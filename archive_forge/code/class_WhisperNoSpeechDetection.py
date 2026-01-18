import inspect
import math
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union
import numpy as np
import torch
from ..utils import add_start_docstrings
from ..utils.logging import get_logger
class WhisperNoSpeechDetection(LogitsProcessor):
    """This processor can be used to detect silence when using Whisper. It should take as input unprocessed logits to follow the original implementation"""

    def __init__(self, no_speech_token: int, begin_index: int, scores_is_logprobs: bool=False):
        self.no_speech_token = no_speech_token
        self.start_of_trans_offset = begin_index
        self.begin_index = begin_index
        self._no_speech_prob = [0.0]
        self.is_scores_logprobs = scores_is_logprobs
        self.model = None
        self.inputs = None

    def set_model(self, model):
        self.model = model

    def set_inputs(self, inputs):
        self.inputs = {**self.model.prepare_inputs_for_generation(**inputs), **inputs}
        self.inputs['input_features'] = self.inputs.pop('inputs')

    @property
    def no_speech_prob(self):
        return self._no_speech_prob

    def set_begin_index(self, begin_index):
        self.begin_index = begin_index

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if input_ids.shape[1] == self.begin_index:
            if self.start_of_trans_offset > 1:
                with torch.no_grad():
                    logits = self.model(**self.inputs).logits
                no_speech_index = self.begin_index - self.start_of_trans_offset
                no_speech_scores = logits[:, no_speech_index]
            else:
                no_speech_scores = scores
            if self.is_scores_logprobs:
                probs = no_speech_scores.exp()
            else:
                probs = no_speech_scores.float().softmax(dim=-1)
            self._no_speech_prob = probs[:, self.no_speech_token]
        return scores