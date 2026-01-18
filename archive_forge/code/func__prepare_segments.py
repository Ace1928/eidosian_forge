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
def _prepare_segments(prompt_ids, batch_size, generation_config):
    if prompt_ids is not None and generation_config.prompt_condition_type == 'first-segment':
        prev_sot_token_id = getattr(generation_config, 'prev_sot_token_id', None)
        prompt_ids = prompt_ids[1:] if prompt_ids[0] == prev_sot_token_id else prompt_ids
        current_segments = [[{'tokens': prompt_ids}] for _ in range(batch_size)]
    else:
        current_segments = [[] for _ in range(batch_size)]
    return current_segments