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
def _setup_no_speech_detection(logits_processor, segment_input, decoder_input_ids, kwargs):
    set_inputs = _get_attr_from_logit_processors(logits_processor, WhisperNoSpeechDetection, 'set_inputs')
    extra_kwargs = {k: v for k, v in kwargs.items() if torch.is_tensor(v)}
    set_inputs({'inputs': segment_input, 'decoder_input_ids': decoder_input_ids, **extra_kwargs})