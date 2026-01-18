import inspect
import math
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union
import numpy as np
import torch
from ..utils import add_start_docstrings
from ..utils.logging import get_logger
class SequenceBiasLogitsProcessor(LogitsProcessor):
    """
    [`LogitsProcessor`] that applies an additive bias on sequences. The bias is applied to the last token of a sequence
    when the next generated token can complete it. Consequently, to take the most of biasing sequences with more than
    one token, consider using beam methods (to gracefully work around partially completed sequences that have a
    negative bias) and applying the bias to their prefixes (to ensure the bias is applied earlier).

    <Tip>

    In order to get the token ids of the sequences that you want to bias, make sure to set `add_prefix_space=True` when
    initializing the tokenizer, and use `tokenizer(bad_words, add_special_tokens=False).input_ids`. The
    `add_prefix_space` argument is only supported for some slow tokenizers, as fast tokenizers' prefixing behaviours
    come from `pre tokenizers`. Read more [here](https://huggingface.co/docs/tokenizers/api/pre-tokenizers).

    </Tip>

    Args:
        sequence_bias (`Dict[Tuple[int], float]`):
            Dictionary that maps a sequence of tokens to its bias term. Positive biases increase the odds of the
            sequence being selected, while negative biases do the opposite. If a sequence has a length of 1, its bias
            will always be applied. Otherwise, the bias will only be applied if the sequence in question is about to be
            completed (in the token selection step after this processor is applied).

    Examples:

    ```python
    >>> from transformers import AutoTokenizer, AutoModelForCausalLM

    >>> model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
    >>> tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    >>> inputs = tokenizer(["The full name of Donald is Donald"], return_tensors="pt")

    >>> summary_ids = model.generate(inputs["input_ids"], max_new_tokens=4)
    >>> print(tokenizer.batch_decode(summary_ids, skip_special_tokens=True)[0])
    The full name of Donald is Donald J. Trump Jr

    >>> # Now let's control generation through a bias. Please note that the tokenizer is initialized differently!
    >>> tokenizer_with_prefix_space = AutoTokenizer.from_pretrained("openai-community/gpt2", add_prefix_space=True)


    >>> def get_tokens_as_tuple(word):
    ...     return tuple(tokenizer_with_prefix_space([word], add_special_tokens=False).input_ids[0])


    >>> # If we add a negative bias without beam search, it may become "stuck" in a prefix without good continuations
    >>> sequence_bias = {get_tokens_as_tuple("Trump"): -10.0}
    >>> biased_ids = model.generate(inputs["input_ids"], max_new_tokens=4, sequence_bias=sequence_bias)
    >>> print(tokenizer.batch_decode(biased_ids, skip_special_tokens=True)[0])
    The full name of Donald is Donald J. Donald,

    >>> biased_ids = model.generate(inputs["input_ids"], max_new_tokens=4, num_beams=4, sequence_bias=sequence_bias)
    >>> print(tokenizer.batch_decode(biased_ids, skip_special_tokens=True)[0])
    The full name of Donald is Donald Rumsfeld,

    >>> # We can also add a positive bias to nudge the model towards specific tokens or continuations
    >>> sequence_bias = {get_tokens_as_tuple("Donald Duck"): 10.0}
    >>> biased_ids = model.generate(inputs["input_ids"], max_new_tokens=4, num_beams=4, sequence_bias=sequence_bias)
    >>> print(tokenizer.batch_decode(biased_ids, skip_special_tokens=True)[0])
    The full name of Donald is Donald Duck.
    ```
    """

    def __init__(self, sequence_bias: Dict[Tuple[int], float]):
        self.sequence_bias = sequence_bias
        self._validate_arguments()
        self.length_1_bias = None
        self.prepared_bias_variables = False

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if not self.prepared_bias_variables:
            self._prepare_bias_variables(scores)
        bias = torch.zeros_like(scores)
        bias += self.length_1_bias
        for sequence_ids, sequence_bias in self.sequence_bias.items():
            if len(sequence_ids) == 1:
                continue
            if len(sequence_ids) > input_ids.shape[1]:
                continue
            prefix_length = len(sequence_ids) - 1
            last_token = sequence_ids[-1]
            matching_rows = torch.eq(input_ids[:, -prefix_length:], torch.tensor(sequence_ids[:-1], dtype=input_ids.dtype, device=input_ids.device)).prod(dim=1)
            bias[:, last_token] += torch.where(matching_rows.bool(), torch.tensor(sequence_bias, device=input_ids.device), torch.tensor(0.0, device=input_ids.device))
        scores = scores + bias
        return scores

    def _prepare_bias_variables(self, scores: torch.FloatTensor):
        vocabulary_size = scores.shape[-1]
        invalid_biases = []
        for sequence_ids in self.sequence_bias:
            for token_id in sequence_ids:
                if token_id >= vocabulary_size:
                    invalid_biases.append(token_id)
        if len(invalid_biases) > 0:
            raise ValueError(f'The model vocabulary size is {vocabulary_size}, but the following tokens were being biased: {invalid_biases}')
        self.length_1_bias = torch.zeros((vocabulary_size,), dtype=torch.float).to(scores.device)
        for sequence_ids, bias in self.sequence_bias.items():
            if len(sequence_ids) == 1:
                self.length_1_bias[sequence_ids[-1]] = bias
        self.prepared_bias_variables = True

    def _validate_arguments(self):
        sequence_bias = self.sequence_bias
        if not isinstance(sequence_bias, dict) or len(sequence_bias) == 0:
            raise ValueError(f'`sequence_bias` has to be a non-empty dictionary, but is {sequence_bias}.')
        if any((not isinstance(sequence_ids, tuple) for sequence_ids in sequence_bias.keys())):
            raise ValueError(f'`sequence_bias` has to be a dict with tuples as keys, but is {sequence_bias}.')
        if any((any((not isinstance(token_id, (int, np.integer)) or token_id < 0 for token_id in sequence_ids)) or len(sequence_ids) == 0 for sequence_ids in sequence_bias.keys())):
            raise ValueError(f'Each key in `sequence_bias` has to be a non-empty tuple of positive integers, but is {sequence_bias}.')
        if any((not isinstance(bias, float) for bias in sequence_bias.values())):
            raise ValueError(f'`sequence_bias` has to be a dict with floats as values, but is {sequence_bias}.')