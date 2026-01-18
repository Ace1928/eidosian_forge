import inspect
import math
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union
import numpy as np
import torch
from ..utils import add_start_docstrings
from ..utils.logging import get_logger
class NoBadWordsLogitsProcessor(SequenceBiasLogitsProcessor):
    """
    [`LogitsProcessor`] that enforces that specified sequences will never be selected.

    <Tip>

    In order to get the token ids of the words that should not appear in the generated text, make sure to set
    `add_prefix_space=True` when initializing the tokenizer, and use `tokenizer(bad_words,
    add_special_tokens=False).input_ids`. The `add_prefix_space` argument is only supported for some slow tokenizers,
    as fast tokenizers' prefixing behaviours come from `pre tokenizers`. Read more
    [here](https://huggingface.co/docs/tokenizers/api/pre-tokenizers).

    </Tip>

    Args:
        bad_words_ids (`List[List[int]]`):
            List of list of token ids that are not allowed to be generated.
        eos_token_id (`Union[int, List[int]]`):
            The id of the *end-of-sequence* token. Optionally, use a list to set multiple *end-of-sequence* tokens.

    Examples:

    ```python
    >>> from transformers import AutoTokenizer, AutoModelForCausalLM

    >>> model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
    >>> tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    >>> inputs = tokenizer(["In a word, the cake is a"], return_tensors="pt")

    >>> output_ids = model.generate(inputs["input_ids"], max_new_tokens=5, pad_token_id=tokenizer.eos_token_id)
    >>> print(tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0])
    In a word, the cake is a bit of a mess.

    >>> # Now let's take the bad words out. Please note that the tokenizer is initialized differently
    >>> tokenizer_with_prefix_space = AutoTokenizer.from_pretrained("openai-community/gpt2", add_prefix_space=True)


    >>> def get_tokens_as_list(word_list):
    ...     "Converts a sequence of words into a list of tokens"
    ...     tokens_list = []
    ...     for word in word_list:
    ...         tokenized_word = tokenizer_with_prefix_space([word], add_special_tokens=False).input_ids[0]
    ...         tokens_list.append(tokenized_word)
    ...     return tokens_list


    >>> bad_words_ids = get_tokens_as_list(word_list=["mess"])
    >>> output_ids = model.generate(
    ...     inputs["input_ids"], max_new_tokens=5, bad_words_ids=bad_words_ids, pad_token_id=tokenizer.eos_token_id
    ... )
    >>> print(tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0])
    In a word, the cake is a bit of a surprise.
    ```
    """

    def __init__(self, bad_words_ids: List[List[int]], eos_token_id: Union[int, List[int]]):
        self.bad_word_ids = bad_words_ids
        self._validate_arguments()
        if eos_token_id is None:
            eos_token_id = []
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        bad_words_ids = list(filter(lambda bad_token_seq: all((bad_token_seq != [i] for i in eos_token_id)), bad_words_ids))
        sequence_bias = {tuple(sequence): float('-inf') for sequence in bad_words_ids}
        super().__init__(sequence_bias=sequence_bias)

    def _validate_arguments(self):
        bad_words_ids = self.bad_word_ids
        if not isinstance(bad_words_ids, list) or len(bad_words_ids) == 0:
            raise ValueError(f'`bad_words_ids` has to be a non-empty list, but is {bad_words_ids}.')
        if any((not isinstance(bad_word_ids, list) for bad_word_ids in bad_words_ids)):
            raise ValueError(f'`bad_words_ids` has to be a list of lists, but is {bad_words_ids}.')
        if any((any((not isinstance(token_id, (int, np.integer)) or token_id < 0 for token_id in bad_word_ids)) for bad_word_ids in bad_words_ids)):
            raise ValueError(f'Each list in `bad_words_ids` has to be a list of positive integers, but is {bad_words_ids}.')