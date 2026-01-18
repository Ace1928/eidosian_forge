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
@torch.no_grad()
def _sample(self, music_tokens, labels, sample_levels, metas=None, chunk_size=32, sampling_temperature=0.98, lower_batch_size=16, max_batch_size=16, sample_length_in_seconds=24, compute_alignments=False, sample_tokens=None, offset=0, save_results=True, sample_length=None) -> List[torch.LongTensor]:
    """
        Core sampling function used to generate music tokens. Iterates over the provided list of levels, while saving
        the generated raw audio at each step.

        Args:
            music_tokens (`List[torch.LongTensor]`):
                A sequence of music tokens of length `self.levels` which will be used as context to continue the
                sampling process. Should have `self.levels` tensors, each corresponding to the generation at a certain
                level.
            labels (`List[torch.LongTensor]`):
                List of length `n_sample`, and shape `(self.levels, 4 + self.config.max_nb_genre +
                lyric_sequence_length)` metadata such as `artist_id`, `genre_id` and the full list of lyric tokens
                which are used to condition the generation.
            sample_levels (`List[int]`):
                List of the desired levels at which the sampling will be done. A level is equivalent to the index of
                the prior in the list of priors
            metas (`List[Any]`, *optional*):
                Metadatas used to generate the `labels`
            chunk_size (`int`, *optional*, defaults to 32):
                Size of a chunk of audio, used to fill up the memory in chuncks to prevent OOM erros. Bigger chunks
                means faster memory filling but more consumption.
            sampling_temperature (`float`, *optional*, defaults to 0.98):
                Temperature used to ajust the randomness of the sampling.
            lower_batch_size (`int`, *optional*, defaults to 16):
                Maximum batch size for the lower level priors
            max_batch_size (`int`, *optional*, defaults to 16):
                Maximum batch size for the top level priors
            sample_length_in_seconds (`int`, *optional*, defaults to 24):
                Desired length of the generation in seconds
            compute_alignments (`bool`, *optional*, defaults to `False`):
                Whether or not to compute the alignment between the lyrics and the audio using the top_prior
            sample_tokens (`int`, *optional*):
                Precise number of tokens that should be sampled at each level. This is mostly useful for running dummy
                experiments
            offset (`int`, *optional*, defaults to 0):
                Audio offset used as conditioning, corresponds to the starting sample in the music. If the offset is
                greater than 0, the lyrics will be shifted take that intoaccount
            save_results (`bool`, *optional*, defaults to `True`):
                Whether or not to save the intermediate results. If `True`, will generate a folder named with the start
                time.
            sample_length (`int`, *optional*):
                Desired length of the generation in samples.

        Returns: torch.Tensor

        Example:

        ```python
        >>> from transformers import AutoTokenizer, JukeboxModel, set_seed
        >>> import torch

        >>> metas = dict(artist="Zac Brown Band", genres="Country", lyrics="I met a traveller from an antique land")
        >>> tokenizer = AutoTokenizer.from_pretrained("openai/jukebox-1b-lyrics")
        >>> model = JukeboxModel.from_pretrained("openai/jukebox-1b-lyrics", min_duration=0).eval()

        >>> labels = tokenizer(**metas)["input_ids"]
        >>> set_seed(0)
        >>> zs = [torch.zeros(1, 0, dtype=torch.long) for _ in range(3)]
        >>> zs = model._sample(zs, labels, [0], sample_length=40 * model.priors[0].raw_to_tokens, save_results=False)
        >>> zs[0]
        tensor([[1853, 1369, 1150, 1869, 1379, 1789,  519,  710, 1306, 1100, 1229,  519,
              353, 1306, 1379, 1053,  519,  653, 1631, 1467, 1229, 1229,   10, 1647,
             1254, 1229, 1306, 1528, 1789,  216, 1631, 1434,  653,  475, 1150, 1528,
             1804,  541, 1804, 1434]])
        ```
        """
    top_prior = self.priors[0]
    if sample_length is not None:
        total_length = sample_length
    else:
        total_length = int(sample_length_in_seconds * self.config.sampling_rate) // top_prior.raw_to_tokens * top_prior.raw_to_tokens
    if sample_levels is None:
        sample_levels = range(len(self.priors))
    self.total_length = total_length
    for level in sample_levels:
        sampling_kwargs = {'temp': 0.99 if level == len(self.priors) - 1 else sampling_temperature, 'chunk_size': chunk_size, 'sample_tokens': sample_tokens}
        total_token_to_sample = total_length // self.priors[level].raw_to_tokens
        hop_length = int(self.config.hop_fraction[level] * self.priors[level].n_ctx)
        max_batch_size = lower_batch_size if level != sample_levels else max_batch_size
        music_tokens = self.sample_level(music_tokens, labels[level], offset, sampling_kwargs, level, total_token_to_sample, hop_length, max_batch_size)
        if save_results:
            self.vqvae.to(music_tokens[level].device)
            with torch.no_grad():
                start_level = len(self.priors) - level - 1
                raw_audio = self.vqvae.decode(music_tokens[:level + 1], start_level=start_level, bs_chunks=music_tokens[level].shape[0])
            logdir = f'jukebox/level_{level}'
            if not os.path.exists(logdir):
                os.makedirs(logdir)
            save_temp_audio(logdir, level, metas=metas, aud=raw_audio.float())
            if compute_alignments and self.priors[0] is not None and (self.priors[0].nb_relevant_lyric_tokens > 0):
                with torch.no_grad():
                    alignments = get_alignment(music_tokens, labels[0], self.priors[0], self.config)
                torch.save({'alignments': alignments}, f'{logdir}/lyric_alignments.pt')
    return music_tokens