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
def _set_return_outputs(return_dict_in_generate, return_token_timestamps, is_shortform, logprob_threshold, generation_config):
    if return_dict_in_generate is None:
        return_dict_in_generate = generation_config.return_dict_in_generate
    generation_config.return_token_timestamps = return_token_timestamps
    if return_token_timestamps:
        return_dict_in_generate = True
        generation_config.output_attentions = True
        generation_config.output_scores = True
    if not is_shortform and logprob_threshold is not None:
        return_dict_in_generate = True
        generation_config.output_scores = True
    generation_config.return_dict_in_generate = return_dict_in_generate