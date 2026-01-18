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
def _maybe_warn_unused_inputs(condition_on_prev_tokens, temperature, compression_ratio_threshold, logprob_threshold, no_speech_threshold, total_input_frames):
    warning_prefix = f'Audio input consists of only {total_input_frames}. Short-form transcription is activated.{{}}, but will be ignored.'
    if condition_on_prev_tokens is not None:
        logger.warn(warning_prefix.format(f'condition_on_prev_tokens is set to {condition_on_prev_tokens}'))
    if compression_ratio_threshold is not None:
        logger.warn(warning_prefix.format(f'compression_ratio_threshold is set to {compression_ratio_threshold}'))
    if logprob_threshold is not None:
        logger.warn(warning_prefix.format(f'logprob_threshold is set to {logprob_threshold}'))
    if no_speech_threshold is not None:
        logger.warn(warning_prefix.format(f'no_speech_threshold is set to {no_speech_threshold}'))
    if isinstance(temperature, (list, tuple)):
        raise ValueError(f'Audio input consists of only {total_input_frames}. Short-form transcription is activated.temperature cannot be set to {temperature} which can only be used for temperature fallback for long-form generation. Make sure to set `temperature` to a float value or `None` for short-form generation.')