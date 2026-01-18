import inspect
import math
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union
import numpy as np
import torch
from ..utils import add_start_docstrings
from ..utils.logging import get_logger
class MinLengthLogitsProcessor(LogitsProcessor):
    """
    [`LogitsProcessor`] enforcing a min-length by setting EOS probability to 0. Note that, for decoder-only models
    like most LLMs, the length includes the prompt.

    Args:
        min_length (`int`):
            The minimum length below which the score of `eos_token_id` is set to `-float("Inf")`.
        eos_token_id (`Union[int, List[int]]`):
            The id of the *end-of-sequence* token. Optionally, use a list to set multiple *end-of-sequence* tokens.

    Examples:

    ```python
    >>> from transformers import AutoModelForCausalLM, AutoTokenizer

    >>> tokenizer = AutoTokenizer.from_pretrained("bigscience/bloomz-560m")
    >>> model = AutoModelForCausalLM.from_pretrained("bigscience/bloomz-560m")

    >>> inputs = tokenizer("A number:", return_tensors="pt")
    >>> gen_out = model.generate(**inputs)
    >>> print(tokenizer.batch_decode(gen_out, skip_special_tokens=True)[0])
    A number: one

    >>> # setting `min_length` to a value smaller than the uncontrolled output length has no impact
    >>> gen_out = model.generate(**inputs, min_length=3)
    >>> print(tokenizer.batch_decode(gen_out, skip_special_tokens=True)[0])
    A number: one

    >>> # setting a larger `min_length` will force the model to generate beyond its natural ending point, which is not
    >>> # necessarily incorrect
    >>> gen_out = model.generate(**inputs, min_length=10)
    >>> print(tokenizer.batch_decode(gen_out, skip_special_tokens=True)[0])
    A number: one thousand, nine hundred and ninety-four
    ```
    """

    def __init__(self, min_length: int, eos_token_id: Union[int, List[int]]):
        if not isinstance(min_length, int) or min_length < 0:
            raise ValueError(f'`min_length` has to be a non-negative integer, but is {min_length}')
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        if not all((isinstance(i, int) for i in eos_token_id)) or any((i < 0 for i in eos_token_id)):
            logger.warning(f'`eos_token_id` has to be a list of positive integers, but is {eos_token_id}')
        self.min_length = min_length
        self.eos_token_id = eos_token_id

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        cur_len = input_ids.shape[-1]
        if cur_len < self.min_length:
            for i in self.eos_token_id:
                scores[:, i] = -float('inf')
        return scores