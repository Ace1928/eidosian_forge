import inspect
import math
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union
import numpy as np
import torch
from ..utils import add_start_docstrings
from ..utils.logging import get_logger
class EncoderRepetitionPenaltyLogitsProcessor(LogitsProcessor):
    """
    [`LogitsProcessor`] that works similarly to [`RepetitionPenaltyLogitsProcessor`], but with an *inverse* penalty
    that is applied to the tokens present in the prompt. In other words, a penalty above 1.0 increases the odds of
    selecting tokens that were present in the prompt.

    It was designed to avoid hallucination in input-grounded tasks, like summarization. Although originally intended
    for encoder-decoder models, it can also be used with decoder-only models like LLMs.

    Args:
        penalty (`float`):
            The parameter for repetition penalty. 1.0 means no penalty. Above 1.0 rewards prompt tokens. Between 0.0
            and 1.0 penalizes prompt tokens.
        encoder_input_ids (`torch.LongTensor`):
            The encoder_input_ids that should be repeated within the decoder ids.

    Examples:

    ```python
    >>> from transformers import AutoModelForCausalLM, AutoTokenizer

    >>> tokenizer = AutoTokenizer.from_pretrained("bigscience/bloomz-560m")
    >>> model = AutoModelForCausalLM.from_pretrained("bigscience/bloomz-560m")

    >>> inputs = tokenizer(["Alice and Bob. The third member's name was"], return_tensors="pt")
    >>> gen_out = model.generate(**inputs)
    >>> print(tokenizer.batch_decode(gen_out, skip_special_tokens=True)[0])
    Alice and Bob. The third member's name was not mentioned.

    >>> # With the `encoder_repetition_penalty` argument we can trigger this logits processor in `generate`, which can
    >>> # promote the use of prompt tokens ("Bob" in this example)
    >>> gen_out = model.generate(**inputs, encoder_repetition_penalty=1.2)
    >>> print(tokenizer.batch_decode(gen_out, skip_special_tokens=True)[0])
    Alice and Bob. The third member's name was Bob. The third member's name was Bob.
    ```
    """

    def __init__(self, penalty: float, encoder_input_ids: torch.LongTensor):
        if not isinstance(penalty, float) or not penalty > 0:
            raise ValueError(f'`penalty` has to be a strictly positive float, but is {penalty}')
        self.penalty = 1 / penalty
        self.encoder_input_ids = encoder_input_ids

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        score = torch.gather(scores, 1, self.encoder_input_ids)
        score = torch.where(score < 0, score * self.penalty, score / self.penalty)
        scores.scatter_(1, self.encoder_input_ids, score)
        return scores