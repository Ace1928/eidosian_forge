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
@staticmethod
def _get_input_segment(input_features, seek, seek_num_frames, num_segment_frames, cur_bsz, batch_idx_map):
    segment_input = []
    for i in range(cur_bsz):
        prev_i = batch_idx_map[i]
        segment_input_slice = input_features[i:i + 1, :, seek[prev_i]:seek[prev_i] + seek_num_frames[prev_i]]
        if segment_input_slice.shape[-1] < num_segment_frames:
            segment_input_slice = F.pad(segment_input_slice, pad=(0, num_segment_frames - segment_input_slice.shape[-1]))
        segment_input.append(segment_input_slice)
    segment_input = torch.cat(segment_input, dim=0)
    return segment_input