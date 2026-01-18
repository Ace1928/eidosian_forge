import inspect
import math
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union
import numpy as np
import torch
from ..utils import add_start_docstrings
from ..utils.logging import get_logger
class BarkEosPrioritizerLogitsProcessor(LogitsProcessor):
    """This processor ensures that the EOS token is selected if its probability is greater than the `min_eos_p`.

    <Tip warning={true}>

    This logits processor is exclusively compatible with
    [Bark](https://huggingface.co/docs/transformers/en/model_doc/bark). See the model documentation for examples.

    </Tip>

    Args:
        eos_token_id (`Union[int, List[int]]`):
            The id of the *end-of-sequence* token. Optionally, use a list to set multiple *end-of-sequence* tokens.
        min_eos_p (`float`, *optional*):
            Minimum end of speech threshold.
    """

    def __init__(self, eos_token_id: Union[int, List[int]], min_eos_p: float):
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        self.eos_token_id = eos_token_id
        if min_eos_p is not None and min_eos_p <= 0:
            raise ValueError(f'`min_eos_p` has to be a positive float, but is {min_eos_p}')
        self.min_eos_p = min_eos_p

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if self.min_eos_p:
            probs = torch.nn.functional.softmax(scores.float(), dim=-1)
            early_stop_scores = torch.ones_like(scores) * -float('inf')
            early_stop_scores[:, self.eos_token_id] = scores[:, self.eos_token_id]
            do_early_stop = probs[:, self.eos_token_id] > self.min_eos_p
            do_early_stop = torch.any(do_early_stop, dim=1, keepdim=True)
            scores = torch.where(do_early_stop, early_stop_scores, scores)
        return scores