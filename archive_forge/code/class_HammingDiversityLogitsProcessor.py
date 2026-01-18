import inspect
import math
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union
import numpy as np
import torch
from ..utils import add_start_docstrings
from ..utils.logging import get_logger
class HammingDiversityLogitsProcessor(LogitsProcessor):
    """
    [`LogitsProcessor`] that enforces diverse beam search.

    Note that this logits processor is only effective for [`PreTrainedModel.group_beam_search`]. See [Diverse Beam
    Search: Decoding Diverse Solutions from Neural Sequence Models](https://arxiv.org/pdf/1610.02424.pdf) for more
    details.

    Traditional beam search often generates very similar sequences across different beams.
    `HammingDiversityLogitsProcessor` addresses this by penalizing beams that generate tokens already chosen by other
    beams in the same time step.

    Args:
        diversity_penalty (`float`):
            This value is subtracted from a beam's score if it generates a token same as any beam from other group at a
            particular time. A higher `diversity_penalty` will enforce greater diversity among the beams. Adjusting
            this value can help strike a balance between diversity and natural likelihood.
        num_beams (`int`):
            Number of beams for beam search. 1 means no beam search.
        num_beam_groups (`int`):
            Number of groups to divide `num_beams` into in order to ensure diversity among different groups of beams.
            [this paper](https://arxiv.org/pdf/1610.02424.pdf) for more details.

    Examples:

    ```python
    >>> from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    >>> import torch

    >>> # Initialize the model and tokenizer
    >>> tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-base")
    >>> model = AutoModelForSeq2SeqLM.from_pretrained("google-t5/t5-base")

    >>> # A long text about the solar system
    >>> text = (
    ...     "The Solar System is a gravitationally bound system comprising the Sun and the objects that orbit it, "
    ...     "either directly or indirectly. Of the objects that orbit the Sun directly, the largest are the eight "
    ...     "planets, with the remainder being smaller objects, such as the five dwarf planets and small Solar System "
    ...     "bodies. The Solar System formed 4.6 billion years ago from the gravitational collapse of a giant "
    ...     "interstellar molecular cloud."
    ... )
    >>> inputs = tokenizer("summarize: " + text, return_tensors="pt")

    >>> # Generate diverse summary
    >>> outputs_diverse = model.generate(
    ...     **inputs,
    ...     num_beam_groups=2,
    ...     diversity_penalty=10.0,
    ...     max_length=100,
    ...     num_beams=4,
    ...     num_return_sequences=2,
    ... )
    >>> summaries_diverse = tokenizer.batch_decode(outputs_diverse, skip_special_tokens=True)

    >>> # Generate non-diverse summary
    >>> outputs_non_diverse = model.generate(
    ...     **inputs,
    ...     max_length=100,
    ...     num_beams=4,
    ...     num_return_sequences=2,
    ... )
    >>> summary_non_diverse = tokenizer.batch_decode(outputs_non_diverse, skip_special_tokens=True)

    >>> # With `diversity_penalty`, the resulting beams are much more diverse
    >>> print(summary_non_diverse)
    ['the solar system formed 4.6 billion years ago from the collapse of a giant interstellar molecular cloud. of the objects that orbit the Sun directly, the largest are the eight planets.',
    'the Solar System formed 4.6 billion years ago from the collapse of a giant interstellar molecular cloud. of the objects that orbit the Sun directly, the largest are the eight planets.']

    >>> print(summaries_diverse)
    ['the solar system formed 4.6 billion years ago from the collapse of a giant interstellar molecular cloud. of the objects that orbit the Sun directly, the largest are the eight planets.',
    'the solar system formed 4.6 billion years ago from the collapse of a giant interstellar molecular cloud. of the objects that orbit the Sun directly, the largest are the eight planets. the rest of the objects are smaller objects, such as the five dwarf planets and small solar system bodies.']
    ```
    """

    def __init__(self, diversity_penalty: float, num_beams: int, num_beam_groups: int):
        if not isinstance(diversity_penalty, float) or not diversity_penalty > 0.0:
            raise ValueError('`diversity_penalty` should be a float strictly larger than 0.')
        self._diversity_penalty = diversity_penalty
        if not isinstance(num_beams, int) or num_beams < 2:
            raise ValueError('`num_beams` should be an integer strictly larger than 1.')
        self._num_beams = num_beams
        if not isinstance(num_beam_groups, int) or num_beam_groups < 2:
            raise ValueError('`num_beam_groups` should be an integer strictly larger than 1.')
        if num_beam_groups > num_beams:
            raise ValueError('`beam_groups` has to be smaller or equal to `num_beams`.')
        self._num_sub_beams = num_beams // num_beam_groups

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, current_tokens: torch.LongTensor, beam_group_idx: int) -> torch.FloatTensor:
        """
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. [What are input IDs?](../glossary#input-ids)
            scores (`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`):
                Prediction scores of a language modeling head. These can be logits for each vocabulary when not using
                beam search or log softmax for each vocabulary token when using beam search
            current_tokens (`torch.LongTensor` of shape `(batch_size)`):
                Indices of input sequence tokens in the vocabulary, corresponding to the tokens selected by the other
                beam groups in the current generation step.
            beam_group_idx (`int`):
                The index of the beam group currently being processed.

        Return:
            `torch.FloatTensor` of shape `(batch_size, config.vocab_size)`:
                The processed prediction scores.
        """
        batch_size = current_tokens.shape[0] // self._num_beams
        group_start_idx = beam_group_idx * self._num_sub_beams
        group_end_idx = min(group_start_idx + self._num_sub_beams, self._num_beams)
        group_size = group_end_idx - group_start_idx
        vocab_size = scores.shape[-1]
        if group_start_idx == 0:
            return scores
        for batch_idx in range(batch_size):
            previous_group_tokens = current_tokens[batch_idx * self._num_beams:batch_idx * self._num_beams + group_start_idx]
            token_frequency = torch.bincount(previous_group_tokens, minlength=vocab_size).to(scores.device)
            scores[batch_idx * group_size:(batch_idx + 1) * group_size] -= self._diversity_penalty * token_frequency
        return scores