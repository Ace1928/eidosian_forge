import copy
import math
import warnings
import zlib
from typing import Callable, Iterator, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from ...generation.configuration_utils import GenerationConfig
from ...generation.logits_process import (
from ...generation.stopping_criteria import StoppingCriteriaList
from ...modeling_outputs import BaseModelOutput
from ...utils import logging
from .tokenization_whisper import TASK_IDS, TO_LANGUAGE_CODE
def _extract_token_timestamps(self, generate_outputs, alignment_heads, time_precision=0.02, num_frames=None):
    """
        Calculates token-level timestamps using the encoder-decoder cross-attentions and dynamic time-warping (DTW) to
        map each output token to a position in the input audio. If `num_frames` is specified, the encoder-decoder
        cross-attentions will be cropped before applying DTW.

        Returns:
            tensor containing the timestamps in seconds for each predicted token
        """
    cross_attentions = []
    for i in range(self.config.decoder_layers):
        cross_attentions.append(torch.cat([x[i] for x in generate_outputs.cross_attentions], dim=2))
    weights = torch.stack([cross_attentions[l][:, h] for l, h in alignment_heads])
    weights = weights.permute([1, 0, 2, 3])
    weight_length = None
    if 'beam_indices' in generate_outputs:
        weight_length = (generate_outputs.beam_indices != -1).sum(-1).max()
        weights = weights[:, :, :weight_length]
        beam_indices = generate_outputs.beam_indices[:, :weight_length]
        beam_indices = beam_indices.masked_fill(beam_indices == -1, 0)
        weights = torch.stack([torch.index_select(weights[:, :, i, :], dim=0, index=beam_indices[:, i]) for i in range(beam_indices.shape[1])], dim=2)
    input_length = weight_length or cross_attentions[0].shape[2]
    timestamps = torch.zeros_like(generate_outputs.sequences, dtype=torch.float32)[:, :input_length + 1]
    batch_size = timestamps.shape[0]
    if num_frames is not None:
        if len(np.unique(num_frames)) == 1:
            num_frames = num_frames if isinstance(num_frames, int) else num_frames[0]
            weights = weights[..., :num_frames // 2]
        else:
            repeat_time = batch_size if isinstance(num_frames, int) else batch_size // len(num_frames)
            num_frames = np.repeat(num_frames, repeat_time)
    if num_frames is None or isinstance(num_frames, int):
        std = torch.std(weights, dim=-2, keepdim=True, unbiased=False)
        mean = torch.mean(weights, dim=-2, keepdim=True)
        weights = (weights - mean) / std
        weights = _median_filter(weights, self.config.median_filter_width)
        weights = weights.mean(dim=1)
    for batch_idx in range(batch_size):
        if num_frames is not None and isinstance(num_frames, (tuple, list, np.ndarray)):
            matrix = weights[batch_idx, ..., :num_frames[batch_idx] // 2]
            std = torch.std(matrix, dim=-2, keepdim=True, unbiased=False)
            mean = torch.mean(matrix, dim=-2, keepdim=True)
            matrix = (matrix - mean) / std
            matrix = _median_filter(matrix, self.config.median_filter_width)
            matrix = matrix.mean(dim=0)
        else:
            matrix = weights[batch_idx]
        text_indices, time_indices = _dynamic_time_warping(-matrix.cpu().double().numpy())
        jumps = np.pad(np.diff(text_indices), (1, 0), constant_values=1).astype(bool)
        jump_times = time_indices[jumps] * time_precision
        timestamps[batch_idx, 1:] = torch.tensor(jump_times)
    return timestamps