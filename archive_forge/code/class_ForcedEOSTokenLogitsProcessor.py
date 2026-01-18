import inspect
import math
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union
import numpy as np
import torch
from ..utils import add_start_docstrings
from ..utils.logging import get_logger
class ForcedEOSTokenLogitsProcessor(LogitsProcessor):
    """
    [`LogitsProcessor`] that enforces the specified token as the last generated token when `max_length` is reached.

    Args:
        max_length (`int`):
            The maximum length of the sequence to be generated.
        eos_token_id (`Union[int, List[int]]`):
            The id of the token to force as the last generated token when `max_length` is reached. Optionally, use a
            list to set multiple *end-of-sequence* tokens.

    Examples:

    ```python
    >>> from transformers import AutoTokenizer, AutoModelForCausalLM

    >>> model = AutoModelForCausalLM.from_pretrained("distilbert/distilgpt2")
    >>> tokenizer = AutoTokenizer.from_pretrained("distilbert/distilgpt2")

    >>> inputs = tokenizer("A sequence: 1, 2, 3", return_tensors="pt")

    >>> # By default, it continues generating according to the model's logits
    >>> outputs = model.generate(**inputs, max_new_tokens=10)
    >>> print(tokenizer.batch_decode(outputs)[0])
    A sequence: 1, 2, 3, 4, 5, 6, 7, 8

    >>> # `forced_eos_token_id` ensures the generation ends with a EOS token
    >>> outputs = model.generate(**inputs, max_new_tokens=10, forced_eos_token_id=tokenizer.eos_token_id)
    >>> print(tokenizer.batch_decode(outputs)[0])
    A sequence: 1, 2, 3, 4, 5, 6, 7,<|endoftext|>
    ```
    """

    def __init__(self, max_length: int, eos_token_id: Union[int, List[int]]):
        self.max_length = max_length
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        self.eos_token_id = eos_token_id

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        cur_len = input_ids.shape[-1]
        if cur_len == self.max_length - 1:
            num_tokens = scores.shape[1]
            scores[:, [i for i in range(num_tokens) if i not in self.eos_token_id]] = -float('inf')
            for i in self.eos_token_id:
                scores[:, i] = 0
        return scores