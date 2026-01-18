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
def _retrieve_max_frames_and_seek(batch_size, attention_mask, total_input_frames):
    if batch_size > 1 and attention_mask is None:
        raise ValueError('When doing batched long-form audio transcription, make sure to pass an `attention_mask`. You can retrieve the `attention_mask` by doing `processor(audio, ..., return_attention_mask=True)` ')
    elif batch_size > 1:
        max_frames = attention_mask.sum(-1).cpu().to(torch.long)
        seek = torch.zeros((batch_size,), dtype=torch.long)
    else:
        max_frames = torch.ones((1,), dtype=torch.long) * total_input_frames
        seek = torch.zeros((1,), dtype=torch.long)
    return (max_frames, seek)