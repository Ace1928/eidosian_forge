import inspect
import math
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union
import numpy as np
import torch
from ..utils import add_start_docstrings
from ..utils.logging import get_logger
class EncoderNoRepeatNGramLogitsProcessor(LogitsProcessor):
    """
    [`LogitsProcessor`] that works similarly to [`NoRepeatNGramLogitsProcessor`], but applied exclusively to prevent
    the repetition of n-grams present in the prompt.

    It was designed to promote chattiness in a language model, by preventing the generation of n-grams present in
    previous conversation rounds.

    Args:
        encoder_ngram_size (`int`):
            All ngrams of size `ngram_size` can only occur within the encoder input ids.
        encoder_input_ids (`int`):
            The encoder_input_ids that should not be repeated within the decoder ids.

    Examples:

    ```py
    >>> from transformers import AutoTokenizer, AutoModelForCausalLM

    >>> model = AutoModelForCausalLM.from_pretrained("bigscience/bloomz-560m")
    >>> tokenizer = AutoTokenizer.from_pretrained("bigscience/bloomz-560m")

    >>> inputs = tokenizer("Alice: I love cats. What do you love?\\nBob:", return_tensors="pt")

    >>> # With greedy decoding, we see Bob repeating Alice's opinion. If Bob was a chatbot, it would be a poor one.
    >>> outputs = model.generate(**inputs)
    >>> print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])
    Alice: I love cats. What do you love?
    Bob: I love cats. What do you

    >>> # With this logits processor, we can prevent Bob from repeating Alice's opinion.
    >>> outputs = model.generate(**inputs, encoder_no_repeat_ngram_size=2)
    >>> print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])
    Alice: I love cats. What do you love?
    Bob: My cats are very cute.
    ```
    """

    def __init__(self, encoder_ngram_size: int, encoder_input_ids: torch.LongTensor):
        if not isinstance(encoder_ngram_size, int) or encoder_ngram_size <= 0:
            raise ValueError(f'`encoder_ngram_size` has to be a strictly positive integer, but is {encoder_ngram_size}')
        self.ngram_size = encoder_ngram_size
        if len(encoder_input_ids.shape) == 1:
            encoder_input_ids = encoder_input_ids.unsqueeze(0)
        self.batch_size = encoder_input_ids.shape[0]
        self.generated_ngrams = _get_ngrams(encoder_ngram_size, encoder_input_ids, self.batch_size)

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        num_hypos = scores.shape[0]
        num_beams = num_hypos // self.batch_size
        cur_len = input_ids.shape[-1]
        banned_batch_tokens = [_get_generated_ngrams(self.generated_ngrams[hypo_idx // num_beams], input_ids[hypo_idx], self.ngram_size, cur_len) for hypo_idx in range(num_hypos)]
        for i, banned_tokens in enumerate(banned_batch_tokens):
            scores[i, banned_tokens] = -float('inf')
        return scores