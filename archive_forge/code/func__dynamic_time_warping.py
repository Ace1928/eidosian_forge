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
def _dynamic_time_warping(matrix: np.ndarray):
    """
    Measures similarity between two temporal sequences: the input audio and the output tokens. Used to generate
    token-level timestamps.
    """
    output_length, input_length = matrix.shape
    cost = np.ones((output_length + 1, input_length + 1), dtype=np.float32) * np.inf
    trace = -np.ones((output_length + 1, input_length + 1), dtype=np.float32)
    cost[0, 0] = 0
    for j in range(1, input_length + 1):
        for i in range(1, output_length + 1):
            c0 = cost[i - 1, j - 1]
            c1 = cost[i - 1, j]
            c2 = cost[i, j - 1]
            if c0 < c1 and c0 < c2:
                c, t = (c0, 0)
            elif c1 < c0 and c1 < c2:
                c, t = (c1, 1)
            else:
                c, t = (c2, 2)
            cost[i, j] = matrix[i - 1, j - 1] + c
            trace[i, j] = t
    i = trace.shape[0] - 1
    j = trace.shape[1] - 1
    trace[0, :] = 2
    trace[:, 0] = 1
    text_indices = []
    time_indices = []
    while i > 0 or j > 0:
        text_indices.append(i - 1)
        time_indices.append(j - 1)
        if trace[i, j] == 0:
            i -= 1
            j -= 1
        elif trace[i, j] == 1:
            i -= 1
        elif trace[i, j] == 2:
            j -= 1
        else:
            raise RuntimeError(f'Internal error in dynamic time warping. Unexpected trace[{i}, {j}]. Please file a bug report.')
    text_indices = np.array(text_indices)[::-1]
    time_indices = np.array(time_indices)[::-1]
    return (text_indices, time_indices)