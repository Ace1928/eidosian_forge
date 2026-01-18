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
def _retrieve_avg_logprobs(scores, tokens, eos_token_id, temperature):
    rescale_temperature = temperature if temperature > 0.0 else 1
    scores = torch.stack(scores).to(tokens.device)
    if scores.shape[0] > tokens.shape[0]:
        scores = scores[:tokens.shape[0]]
    else:
        tokens = tokens[-scores.shape[0]:]
    logprobs = F.log_softmax((scores * rescale_temperature).float(), dim=-1).to(scores.dtype)
    sum_logprobs = sum((logprobs[i][tokens[i]] * (tokens[i] != eos_token_id) for i in range(logprobs.shape[0])))
    length = (tokens != eos_token_id).sum(-1) if eos_token_id is not None else tokens.shape[0]
    avg_logprobs = sum_logprobs / (length + 1)
    return avg_logprobs