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
def _retrieve_logit_processors(self, generation_config, logits_processor, begin_index, is_shortform, num_beams):
    if generation_config.return_timestamps is True:
        timestamp_processor = WhisperTimeStampLogitsProcessor(generation_config, begin_index=begin_index)
        logits_processor = [timestamp_processor] if logits_processor is None else [timestamp_processor] + logits_processor
    if generation_config.suppress_tokens is not None:
        suppress_tokens_processor = SuppressTokensLogitsProcessor(generation_config.suppress_tokens)
        logits_processor = [suppress_tokens_processor] if logits_processor is None else [suppress_tokens_processor] + logits_processor
        generation_config.suppress_tokens = None
    if generation_config.begin_suppress_tokens is not None:
        begin_suppress_processor = SuppressTokensAtBeginLogitsProcessor(generation_config.begin_suppress_tokens, begin_index=begin_index)
        logits_processor = [begin_suppress_processor] if logits_processor is None else [begin_suppress_processor] + logits_processor
        generation_config.begin_suppress_tokens = None
    if generation_config.no_speech_threshold is not None and (not is_shortform):
        no_speech_detector = WhisperNoSpeechDetection(no_speech_token=generation_config.no_timestamps_token_id - 1, begin_index=begin_index, scores_is_logprobs=num_beams > 1)
        logits_processor = [no_speech_detector] if logits_processor is None else [no_speech_detector] + logits_processor
        no_speech_detector.set_model(self)
    if is_shortform and generation_config.forced_decoder_ids is not None:
        forced_tokens_proc = ForceTokensLogitsProcessor(generation_config.forced_decoder_ids)
        logits_processor = [forced_tokens_proc] if logits_processor is None else logits_processor + [forced_tokens_proc]
        generation_config.forced_decoder_ids = None
    return logits_processor