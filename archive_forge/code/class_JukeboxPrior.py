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
class JukeboxPrior(PreTrainedModel):
    """
    The JukeboxPrior class, which is a wrapper around the various conditioning and the transformer. JukeboxPrior can be
    seen as language models trained on music. They model the next `music token` prediction task. If a (lyric) `encoderù
    is defined, it also models the `next character` prediction on the lyrics. Can be conditionned on timing, artist,
    genre, lyrics and codes from lower-levels Priors.

    Args:
        config (`JukeboxPriorConfig`):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
        level (`int`, *optional*):
            Current level of the Prior. Should be in range `[0,nb_priors]`.
        nb_priors (`int`, *optional*, defaults to 3):
            Total number of priors.
        vqvae_encoder (`Callable`, *optional*):
            Encoding method of the VQVAE encoder used in the forward pass of the model. Passing functions instead of
            the vqvae module to avoid getting the parameters.
        vqvae_decoder (`Callable`, *optional*):
            Decoding method of the VQVAE decoder used in the forward pass of the model. Passing functions instead of
            the vqvae module to avoid getting the parameters.
    """
    config_class = JukeboxPriorConfig

    def _init_weights(self, module):
        init_scale = self.config.init_scale
        if isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02 * init_scale)
        elif isinstance(module, JukeboxConv1D):
            if self.config.zero_out:
                module.weight.data.zero_()
            else:
                module.weight.data.normal_(mean=0.0, std=0.02 * init_scale)
        elif isinstance(module, JukeboxPositionalEmbedding):
            module.pos_emb.data.normal_(mean=0.0, std=0.01 * init_scale)
        elif isinstance(module, JukeboxRangeEmbedding):
            module.emb.weight.data.normal_(mean=0.0, std=0.01 * init_scale)
        elif isinstance(module, JukeboxConditionalAutoregressive) and hasattr(module, 'lm_head'):
            module.lm_head.weight.data.normal_(mean=0.0, std=0.02 * init_scale)
        elif isinstance(module, JukeboxConditionalAutoregressive) and hasattr(module, 'start_token'):
            module.start_token.data.normal_(mean=0.0, std=0.01 * init_scale)
        elif isinstance(module, JukeboxResConv1DBlock) and self.config.zero_out:
            module.conv1d_2.weigth.data.zero_()
            module.conv1d_2.bias.data.zero_()
        if isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def __init__(self, config: JukeboxPriorConfig, level=None, nb_priors=3, vqvae_encoder=None, vqvae_decoder=None):
        super().__init__(config)
        self.vqvae_encoder = vqvae_encoder
        self.vqvae_decoder = vqvae_decoder
        self.levels = nb_priors
        self.level = level if level is not None else config.level
        self.base_model_prefix = f'priors.{self.level}'
        self.n_ctx = config.n_ctx
        self.lyric_conditioning = config.nb_relevant_lyric_tokens > 0
        self.nb_relevant_lyric_tokens = config.nb_relevant_lyric_tokens
        self.encoder_loss_fraction = config.encoder_loss_fraction
        self.audio_conditioning = self.level != 0
        self.cond_level = self.level - 1
        if self.audio_conditioning:
            self.conditioner_blocks = JukeboxMusicTokenConditioner(config, self.level)
        self.metadata_conditioning = config.metadata_conditioning
        if self.metadata_conditioning:
            self.metadata_embedding = JukeboxLabelConditioner(config, include_time_signal=not self.audio_conditioning)
        self.is_encoder_decoder = config.is_encoder_decoder
        if config.is_encoder_decoder:
            self.input_shapes = [config.nb_relevant_lyric_tokens, config.n_ctx]
            self.embed_dim_shift = [0, config.lyric_vocab_size]
            self.width = config.hidden_size
            self.nb_relevant_lyric_tokens = config.nb_relevant_lyric_tokens
            self.prior = JukeboxConditionalAutoregressive(config, n_ctx=config.nb_relevant_lyric_tokens + config.n_ctx, embed_dim=config.lyric_vocab_size + config.music_vocab_size, audio_conditioning=self.audio_conditioning or self.metadata_conditioning, metadata_conditioning=True)
        else:
            encoder_config = config.encoder_config
            if self.nb_relevant_lyric_tokens != 0 and self.lyric_conditioning:
                self.lyric_acts_width = encoder_config.hidden_size
                self.encoder_width = config.hidden_size
                self.encoder_dim = config.lyric_vocab_size
                self.encoder = JukeboxConditionalAutoregressive(encoder_config, n_ctx=self.nb_relevant_lyric_tokens, embed_dim=self.encoder_dim, audio_conditioning=False, metadata_conditioning=False, is_encoder=True)
                self.encoder.proj_in = JukeboxConv1D(encoder_config.hidden_size, config.hidden_size)
                self.encoder.final_layer_norm = JukeboxLayerNorm(config.hidden_size)
                self.encoder.lm_head = nn.Linear(config.hidden_size, config.lyric_vocab_size, bias=False)
            else:
                self.nb_relevant_lyric_tokens = 0
            self.prior = JukeboxConditionalAutoregressive(config, audio_conditioning=self.audio_conditioning or self.metadata_conditioning, metadata_conditioning=self.metadata_conditioning)
        self.next_token_prediction_loss_dims = config.n_ctx
        self.total_loss_dims = self.nb_relevant_lyric_tokens + self.next_token_prediction_loss_dims
        self.downsamples = [stride ** down for stride, down in zip(config.res_strides_t, config.res_downs_t)]
        self.cond_downsample = self.downsamples[self.level] if self.level != 0 else None
        self.raw_to_tokens = np.prod(self.downsamples[:nb_priors - self.level])
        self.sample_length = self.n_ctx * self.raw_to_tokens
        logger.info(f'Level:{self.level}, Cond downsample:{self.cond_downsample}, Raw to tokens:{self.raw_to_tokens}, Sample length:{self.sample_length}')

    def get_metadata(self, labels, start, total_length, offset, get_indices=False):
        metadata = labels.clone()
        metadata[:, 0] = total_length
        metadata[:, 2] = int(self.sample_length)
        metadata[:, 1:2] = int(offset * self.raw_to_tokens) + int(start * self.raw_to_tokens)
        metadata, indices = self.set_metadata_lyric_tokens(metadata)
        if get_indices:
            return (metadata, indices)
        else:
            return metadata

    def set_metadata_lyric_tokens(self, labels):
        """
        Processes the full labels to only retreive the relevant lyric tokens and keep the metadata conditioning tokens.
        """
        if self.nb_relevant_lyric_tokens > 0:
            tokens_list = torch.zeros((labels.shape[0], self.nb_relevant_lyric_tokens), dtype=torch.long, device=labels.device)
            indices_list = []
            for idx in range(labels.shape[0]):
                full_tokens = labels.clone()[:, 4 + self.metadata_embedding.max_nb_genres:]
                total_length, offset, duration = (labels[idx, 0], labels[idx, 1], labels[idx, 2])
                tokens, indices = get_relevant_lyric_tokens(full_tokens, self.nb_relevant_lyric_tokens, total_length, offset, duration)
                tokens_list[idx, :] = tokens
                indices_list.append(indices)
            return (torch.cat((labels[:, :4 + self.metadata_embedding.max_nb_genres], tokens_list), dim=-1), indices_list)
        else:
            return (labels, None)

    def get_music_tokens_conds(self, music_tokens, start, end):
        """
        Extracts current level's conditioning music tokens.
        """
        if self.level != 0:
            music_tokens_cond = music_tokens[self.level - 1]
            music_tokens = music_tokens_cond[:, start // self.cond_downsample:end // self.cond_downsample]
            missing_cond_len = self.n_ctx // self.cond_downsample - music_tokens_cond[-1].shape[-1]
            if missing_cond_len > 0:
                init_cond = torch.zeros(1, missing_cond_len).to(music_tokens_cond.device)
                music_tokens_cond = torch.cat((music_tokens_cond, init_cond), dim=-1).long()
            music_tokens_conds = [music_tokens_cond]
        else:
            music_tokens_conds = None
        return music_tokens_conds

    def prior_preprocess(self, tokens, conds):
        """
        Shifts the input tokens to account for the dictionary merge. The embed_dim_shift give by how much the music
        tokens should be shifted by. It is equal to `lyric_vocab_size`.
        """
        batch_size = tokens[0].shape[0]
        for i in range(len(tokens)):
            tokens[i] = (tokens[i] + int(self.embed_dim_shift[i])).view(batch_size, -1)
        for i in range(len(conds)):
            if conds[i] is None:
                conds[i] = torch.zeros((batch_size, self.input_shapes[i], self.width), dtype=tokens[0].dtype, device=tokens[0].device)
        return (torch.cat(tokens, dim=1), torch.cat(conds, dim=1))

    def prior_postprocess(self, tokens):
        """
        Shifts back the input tokens if the model uses an encoder decoder architecture. As the embedding layer is
        shared, `prior_embed_dim_shift` shifts the music token ids by `lyric_vocab_size`. Only returns the music
        tokens.
        """
        batch_size = tokens.shape[0]
        dims = (self.input_shapes[0], tokens.shape[1] - self.input_shapes[0])
        tokens = list(torch.split(tokens, dims, dim=1))
        for i in range(len(tokens)):
            bins_shift = int(self.embed_dim_shift[i])
            tokens[i] = (tokens[i] - bins_shift).view(batch_size, -1)
            tokens[i] = torch.clamp(tokens[i], min=0)
        return tokens[-1]

    def embed_tokens(self, music_tokens_conds):
        """
        Embeds the upper level music tokens and upsamples them to provide as audio conditioning.
        """
        music_tokens_conds = music_tokens_conds[:self.cond_level + 1]
        audio_conditioning = None
        for music_tokens_cond, conditioner_block in reversed(list(zip(music_tokens_conds, [self.conditioner_blocks]))):
            audio_conditioning = conditioner_block(music_tokens_cond, audio_conditioning)
        return audio_conditioning

    def encode(self, hidden_states, start_level=None, end_level=None, bs_chunks=1):
        """
        Encodes the hidden states (raw audio) using the VQVAE's encoder. Returns latent_states.
        """
        if start_level is None:
            start_level = self.level
        if end_level is None:
            end_level = self.levels
        with torch.no_grad():
            latent_states = self.vqvae_encoder(hidden_states, start_level=start_level, end_level=end_level, bs_chunks=bs_chunks)
        return latent_states

    def decode(self, music_tokens, start_level=None, end_level=None, bs_chunks=1):
        """
        Usamples the sequence of codebook vectors to a raw audio.
        """
        if start_level is None:
            start_level = self.level
        if end_level is None:
            end_level = self.levels
        with torch.no_grad():
            output = self.vqvae_decoder(music_tokens, start_level=start_level, end_level=end_level, bs_chunks=bs_chunks)
        return output

    def get_cond(self, music_tokens_conds, metadata):
        """
        Converts the input tokens to input_embeddings. Splits the lyrics form the rest of the metadata. Lyric tokens
        can be None.
        """
        if metadata is not None:
            n_labels = metadata.shape[1] - self.nb_relevant_lyric_tokens
            metadata, lyric_tokens = (metadata[:, :n_labels], metadata[:, n_labels:])
        else:
            metadata, lyric_tokens = (None, None)
        metadata_conditioning, metadata_pos = self.metadata_embedding(metadata) if self.metadata_conditioning else (None, None)
        audio_conditioning = self.embed_tokens(music_tokens_conds) if self.audio_conditioning else metadata_pos
        return (audio_conditioning, metadata_conditioning, lyric_tokens)

    def sample(self, n_samples, music_tokens=None, music_tokens_conds=None, metadata=None, temp=1.0, top_k=0, top_p=0.0, chunk_size=None, sample_tokens=None):
        """
        Ancestral/Prime sampling a window of tokens using the provided conditioning and metadatas.

        Args:
            n_samples (`int`):
                Number of samples to generate.
            music_tokens (`List[torch.LongTensor]`, *optional*):
                Previously gemerated tokens at the current level. Used as context for the generation.
            music_tokens_conds (`List[torch.FloatTensor]`, *optional*):
                Upper-level music tokens generated by the previous prior model. Is `None` if the generation is not
                conditionned on the upper-level tokens.
            metadata (`List[torch.LongTensor]`, *optional*):
                List containing the metatdata tensor with the artist, genre and the lyric tokens.
            temp (`float`, *optional*, defaults to 1.0):
                Sampling temperature.
            top_k (`int`, *optional*, defaults to 0):
                Top k probabilities used for filtering.
            top_p (`float`, *optional*, defaults to 0.0):
                Top p probabilities used for filtering.
            chunk_size (`int`, *optional*):
                Size of the chunks used to prepare the cache of the transformer.
            sample_tokens (`int`, *optional*):
                Number of tokens to sample.

        """
        no_past_context = music_tokens is None or music_tokens.shape[1] == 0
        name = {True: 'Ancestral', False: 'Primed'}[no_past_context]
        logger.info(f'{name} sampling {n_samples} samples with temp={temp}, top_k={top_k}, top_p={top_p}')
        with torch.no_grad():
            audio_conditioning, metadata_conditioning, lyric_tokens = self.get_cond(music_tokens_conds, metadata)
            if self.is_encoder_decoder:
                if no_past_context:
                    lyric_and_music_tokens, audio_conditioning = self.prior_preprocess([lyric_tokens], [None, audio_conditioning])
                else:
                    lyric_and_music_tokens, audio_conditioning = self.prior_preprocess([lyric_tokens, music_tokens], [None, audio_conditioning])
                if sample_tokens is not None:
                    sample_tokens += self.nb_relevant_lyric_tokens
                music_tokens = self.prior.primed_sample(n_samples, lyric_and_music_tokens, audio_conditioning, metadata_conditioning, temp=temp, top_k=top_k, top_p=top_p, chunk_size=chunk_size, sample_tokens=sample_tokens)
                music_tokens = self.prior_postprocess(music_tokens)
            else:
                last_encoder_hidden_states = self.get_encoder_states(lyric_tokens, sample=True)
                if no_past_context:
                    music_tokens = self.prior.sample(n_samples, audio_conditioning, metadata_conditioning, last_encoder_hidden_states, temp=temp, top_k=top_k, top_p=top_p, sample_tokens=sample_tokens)
                else:
                    music_tokens = self.prior.primed_sample(n_samples, music_tokens, audio_conditioning, metadata_conditioning, last_encoder_hidden_states, temp=temp, top_k=top_k, top_p=top_p, chunk_size=chunk_size, sample_tokens=sample_tokens)
        return music_tokens

    def get_encoder_states(self, lyric_tokens, sample=False):
        """
        Retreive the last hidden_states of the lyric encoder that will be attended to by the decoder. Forwards through
        the lyric encoder.
        """
        if self.nb_relevant_lyric_tokens != 0 and self.lyric_conditioning:
            if sample:
                self.encoder = self.encoder.to(lyric_tokens.device)
            lyric_acts = self.encoder(lyric_tokens, None, None, None)
            lyric_acts = self.encoder.proj_in(lyric_acts)
            last_encoder_hidden_states = self.encoder.final_layer_norm(lyric_acts)
        else:
            last_encoder_hidden_states = None
        return last_encoder_hidden_states

    def get_encoder_loss(self, last_encoder_hidden_states, target_lyrics):
        """
        Computes the loss for the lyric encoder: next lyric token prediction.
        """
        if self.lyric_conditioning:
            last_encoder_hidden_states = self.encoder.lm_head(last_encoder_hidden_states)
            encoder_loss = nn.functional.cross_entropy(last_encoder_hidden_states.view(-1, self.encoder_dim), target_lyrics.view(-1)) / np.log(2.0)
        else:
            encoder_loss = torch.tensor(0.0, device=last_encoder_hidden_states.device)
        return encoder_loss

    def forward_tokens(self, music_tokens, music_tokens_conds=[], metadata=None, get_preds=False, get_attn_weights=False):
        """
        Applies a forward pass using the conditioning tokens. Different from the classic forward as it does not use the
        vqvae's encoding layers.
        """
        if get_attn_weights:
            self.prior.transformer.set_record_attn(get_attn_weights)
        audio_conditioning, metadata_conditioning, lyric_tokens = self.get_cond(music_tokens_conds, metadata)
        if self.is_encoder_decoder:
            tokens, audio_conditioning = self.prior_preprocess([lyric_tokens, music_tokens], [None, audio_conditioning])
            (encoder_loss, next_token_prediction_loss), preds = self.prior(tokens, audio_conditioning, metadata_conditioning, get_sep_loss=True, get_preds=get_preds)
        else:
            last_encoder_hidden_states = self.get_encoder_states(lyric_tokens)
            encoder_loss = self.get_encoder_loss(last_encoder_hidden_states, lyric_tokens)
            next_token_prediction_loss, preds = self.prior(music_tokens, audio_conditioning, metadata_conditioning, last_encoder_hidden_states, get_preds=get_preds)
        loss = self.encoder_loss_fraction * encoder_loss * self.nb_relevant_lyric_tokens / self.total_loss_dims
        loss += next_token_prediction_loss * self.next_token_prediction_loss_dims / self.total_loss_dims
        metrics = {'bpd': next_token_prediction_loss.clone().detach(), 'encoder_loss': encoder_loss.clone().detach(), 'next_token_prediction_loss': next_token_prediction_loss.clone().detach()}
        if get_preds:
            metrics['preds'] = preds.clone().detach()
        if get_attn_weights:
            saved_attn_weights = self.prior.transformer.saved_attn_weights
            self.prior.transformer.set_record_attn(False)
            return saved_attn_weights
        else:
            return (loss, metrics)

    def forward(self, hidden_states: torch.Tensor, metadata: Optional[List[torch.LongTensor]], decode: Optional[bool]=False, get_preds: Optional[bool]=False) -> List[torch.Tensor]:
        """
        Encode the hidden states using the `vqvae` encoder, and then predicts the next token in the `forward_tokens`
        function. The loss is the sum of the `encoder` loss and the `decoder` loss.

        Args:
            hidden_states (`torch.Tensor`):
                Hidden states which should be raw audio
            metadata (`List[torch.LongTensor]`, *optional*):
                List containing the metadata conditioning tensorwith the lyric and the metadata tokens.
            decode (`bool`, *optional*, defaults to `False`):
                Whether or not to decode the encoded to tokens.
            get_preds (`bool`, *optional*, defaults to `False`):
                Whether or not to return the actual predicitons of the model.
        """
        batch_size = hidden_states.shape[0]
        music_tokens, *music_tokens_conds = self.encode(hidden_states, bs_chunks=batch_size)
        loss, metrics = self.forward_tokens(music_tokens=music_tokens, music_tokens_conds=music_tokens_conds, metadata=metadata, get_preds=get_preds)
        if decode:
            dequantised_states = self.decode([music_tokens, *music_tokens_conds])
        else:
            dequantised_states = None
        return (dequantised_states, loss, metrics)