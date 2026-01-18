import math
import os
from typing import List, Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import LayerNorm as FusedLayerNorm
from ...activations import ACT2FN
from ...modeling_utils import PreTrainedModel
from ...utils import add_start_docstrings, logging
from ...utils.logging import tqdm
from .configuration_jukebox import ATTENTION_PATTERNS, JukeboxConfig, JukeboxPriorConfig, JukeboxVQVAEConfig
@add_start_docstrings('The Hierarchical VQ-VAE model used in Jukebox. This model follows the Hierarchical VQVAE paper from [Will Williams, Sam\nRinger, Tom Ash, John Hughes, David MacLeod, Jamie Dougherty](https://arxiv.org/abs/2002.08111).\n\n    ', JUKEBOX_START_DOCSTRING)
class JukeboxVQVAE(PreTrainedModel):
    config_class = JukeboxVQVAEConfig
    base_model_prefix = 'vqvae'

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02 * self.config.init_scale)
        elif isinstance(module, JukeboxConv1D):
            if self.config.zero_out:
                module.weight.data.zero_()
            else:
                module.weight.data.normal_(mean=0.0, std=0.02 * self.config.init_scale)
        elif isinstance(module, JukeboxResConv1DBlock) and self.config.zero_out:
            module.conv1d_2.weight.data.zero_()
            module.conv1d_2.bias.data.zero_()
        if isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def __init__(self, config: JukeboxVQVAEConfig):
        super().__init__(config)
        downs_t = config.res_downs_t
        strides_t = config.res_strides_t
        if not config.sample_length:
            downsamples = [stride ** down for stride, down in zip(strides_t, downs_t)]
            top_raw_to_tokens = np.prod(downsamples)
            config.sample_length = config.sample_length_in_seconds * config.sampling_rate // top_raw_to_tokens * top_raw_to_tokens
            config.sample_length = config.sample_length.astype(int)
        self.nb_discrete_codes = config.nb_discrete_codes
        self.commit = config.commit
        self.sample_length = config.sample_length
        self.downsamples = [stride ** down for stride, down in zip(strides_t, downs_t)]
        self.hop_lengths = np.cumprod(self.downsamples)
        self.levels = levels = config.levels
        self.music_tokens_shapes = [int(self.sample_length // self.hop_lengths[-level - 1]) for level in range(levels)]
        self.multipliers = config.multipliers if config.multipliers is not None else [1] * levels
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for level in range(levels):
            width = config.res_conv_width * self.multipliers[level]
            depth = config.res_conv_depth * self.multipliers[level]
            self.encoders.append(JukeboxEncoder(config, width, depth, level + 1, downs_t[:level + 1], strides_t[:level + 1]))
            self.decoders.append(JukeboxDecoder(config, width, depth, level + 1, downs_t[:level + 1], strides_t[:level + 1]))
        self.bottleneck = JukeboxBottleneck(config, levels)

    def _decode(self, music_tokens, start_level=0, end_level=None):
        if end_level is None:
            end_level = self.levels
        latent_states = self.bottleneck.decode(music_tokens, start_level=start_level, end_level=end_level)
        decoder, dequantised_state = (self.decoders[start_level], latent_states[0:1])
        dequantised_state = decoder(dequantised_state, all_levels=False)
        dequantised_state = dequantised_state.permute(0, 2, 1)
        return dequantised_state

    def decode(self, music_tokens, start_level=0, end_level=None, bs_chunks=1) -> torch.Tensor:
        """
        Transforms the input `music_tokens` to their `raw_audio` representation.

        Args:
            music_tokens (`torch.LongTensor`):
                Tensor of music tokens which will be decoded to raw audio by using the codebook. Each music token
                should be an index to a corresponding `code` vector in the codebook.
            start_level (`int`, *optional*):
                Level at which the decoding process will start. Default to 0.
            end_level (`int`, *optional*):
                Level at which the decoding process will start. Default to None.
            bs_chunks (int, *optional*):
                Number of chunks to process at the same time.
        """
        token_chunks = [torch.chunk(token, bs_chunks, dim=0) for token in music_tokens]
        dequantised_states = []
        for i in range(bs_chunks):
            music_tokens_i = [chunks[i] for chunks in token_chunks]
            dequantised_state = self._decode(music_tokens_i, start_level=start_level, end_level=end_level)
            dequantised_states.append(dequantised_state)
        return torch.cat(dequantised_states, dim=0)

    def _encode(self, raw_audio, start_level=0, end_level=None):
        if end_level is None:
            end_level = self.levels
        input_audio = raw_audio.permute(0, 2, 1).float()
        latent_states = []
        for level in range(self.levels):
            encoder = self.encoders[level]
            latent_state = encoder(input_audio)
            latent_states.append(latent_state[-1])
        music_tokens = self.bottleneck.encode(latent_states)
        return music_tokens[start_level:end_level]

    def encode(self, input_audio, start_level=0, end_level=None, bs_chunks=1):
        """
        Transforms the `input_audio` to a discrete representation made out of `music_tokens`.

        Args:
            input_audio (`torch.Tensor`):
                Raw audio which will be encoded to its discrete representation using the codebook. The closest `code`
                form the codebook will be computed for each sequence of samples.
            start_level (`int`, *optional*, defaults to 0):
                Level at which the encoding process will start. Default to 0.
            end_level (`int`, *optional*):
                Level at which the encoding process will start. Default to None.
            bs_chunks (int, *optional*, defaults to 1):
                Number of chunks of raw audio to process at the same time.
        """
        audio_chunks = torch.chunk(input_audio, bs_chunks, dim=0)
        music_tokens_list = []
        for chunk_i in audio_chunks:
            music_tokens_i = self._encode(chunk_i, start_level=start_level, end_level=end_level)
            music_tokens_list.append(music_tokens_i)
        music_tokens = [torch.cat(music_tokens_level, dim=0) for music_tokens_level in zip(*music_tokens_list)]
        return music_tokens

    def sample(self, n_samples):
        music_tokens = [torch.randint(0, self.nb_discrete_codes, size=(n_samples, *music_tokens_shape), device='cpu') for music_tokens_shape in self.music_tokens_shapes]
        return self.decode(music_tokens)

    def forward(self, raw_audio: torch.FloatTensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the VQ-VAE, encodes the `raw_audio` to latent states, which are then decoded for each level.
        The commit loss, which ensure that the encoder's computed embeddings are close to the codebook vectors, is
        computed.

        Args:
            raw_audio (`torch.FloatTensor`):
                Audio input which will be encoded and decoded.

        Returns:
            `Tuple[torch.Tensor, torch.Tensor]`


        Example:
        ```python
        >>> from transformers import JukeboxVQVAE, set_seed
        >>> import torch

        >>> model = JukeboxVQVAE.from_pretrained("openai/jukebox-1b-lyrics").eval()
        >>> set_seed(0)
        >>> zs = [torch.randint(100, (4, 1))]
        >>> model.decode(zs).shape
        torch.Size([4, 8, 1])
        ```
        """
        input_audio = raw_audio.permute(0, 2, 1).float()
        latent_states = []
        for level in range(self.levels):
            encoder = self.encoders[level]
            latent_state = encoder(input_audio)
            latent_states.append(latent_state[-1])
        _, music_tokens, commit_losses, _ = self.bottleneck(latent_states)
        dequantised_states = []
        for level in range(self.levels):
            decoder = self.decoders[level]
            dequantised_state = decoder(music_tokens[level:level + 1], all_levels=False)
            dequantised_states.append(dequantised_state.permute(0, 2, 1))
        commit_loss = sum(commit_losses)
        loss = self.commit * commit_loss
        return (dequantised_states, loss)