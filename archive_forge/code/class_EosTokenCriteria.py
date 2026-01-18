import time
import warnings
from abc import ABC
from copy import deepcopy
from typing import List, Optional, Union
import torch
from ..utils import add_start_docstrings, logging
class EosTokenCriteria(StoppingCriteria):
    """
    This class can be used to stop generation whenever the "end-of-sequence" token is generated.
    By default, it uses the `model.generation_config.eos_token_id`.

    Args:
        eos_token_id (`Union[int, List[int]]`):
            The id of the *end-of-sequence* token. Optionally, use a list to set multiple *end-of-sequence* tokens.
    """

    def __init__(self, eos_token_id: Union[int, List[int]]):
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        self.eos_token_id = torch.tensor(eos_token_id)

    @add_start_docstrings(STOPPING_CRITERIA_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> torch.BoolTensor:
        is_done = torch.isin(input_ids[:, -1], self.eos_token_id.to(input_ids.device))
        return is_done