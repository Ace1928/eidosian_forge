import inspect
import jax
import jax.lax as lax
import jax.numpy as jnp
from ..utils import add_start_docstrings
from ..utils.logging import get_logger
class FlaxSuppressTokensAtBeginLogitsProcessor(FlaxLogitsProcessor):
    """
    [`FlaxLogitsProcessor`] supressing a list of tokens as soon as the `generate` function starts generating using
    `begin_index` tokens. This should ensure that the tokens defined by `begin_suppress_tokens` are not sampled at the
    begining of the generation.

    Args:
        begin_suppress_tokens (`List[int]`):
            Tokens to not sample.
        begin_index (`int`):
            Index where the tokens are suppressed.
    """

    def __init__(self, begin_suppress_tokens, begin_index):
        self.begin_suppress_tokens = list(begin_suppress_tokens)
        self.begin_index = begin_index

    def __call__(self, input_ids, scores, cur_len: int):
        apply_penalty = 1 - jnp.bool_(cur_len - self.begin_index)
        scores = jnp.where(apply_penalty, scores.at[:, self.begin_suppress_tokens].set(-float('inf')), scores)
        return scores