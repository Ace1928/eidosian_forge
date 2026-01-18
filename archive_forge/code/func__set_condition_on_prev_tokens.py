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
def _set_condition_on_prev_tokens(condition_on_prev_tokens, generation_config):
    condition_on_prev_tokens = condition_on_prev_tokens if condition_on_prev_tokens is not None else getattr(generation_config, 'condition_on_prev_tokens', False)
    generation_config.condition_on_prev_tokens = condition_on_prev_tokens