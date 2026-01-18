import time
import warnings
from abc import ABC
from copy import deepcopy
from typing import Optional
import torch
from ..utils import add_start_docstrings, logging
class MaxLengthCriteria(StoppingCriteria):
    """
    This class can be used to stop generation whenever the full generated number of tokens exceeds `max_length`. Keep
    in mind for decoder-only type of transformers, this will include the initial prompted tokens.

    Args:
        max_length (`int`):
            The maximum length that the output sequence can have in number of tokens.
        max_position_embeddings (`int`, *optional*):
            The maximum model length, as defined by the model's `config.max_position_embeddings` attribute.
    """

    def __init__(self, max_length: int, max_position_embeddings: Optional[int]=None):
        self.max_length = max_length
        self.max_position_embeddings = max_position_embeddings

    @add_start_docstrings(STOPPING_CRITERIA_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        cur_len = input_ids.shape[-1]
        is_done = cur_len >= self.max_length
        if self.max_position_embeddings is not None and (not is_done) and (cur_len >= self.max_position_embeddings):
            logger.warning_once(f"This is a friendly reminder - the current text generation call will exceed the model's predefined maximum length ({self.max_position_embeddings}). Depending on the model, you may observe exceptions, performance degradation, or nothing at all.")
        return is_done