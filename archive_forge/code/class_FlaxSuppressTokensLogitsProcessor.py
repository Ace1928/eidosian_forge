import inspect
import jax
import jax.lax as lax
import jax.numpy as jnp
from ..utils import add_start_docstrings
from ..utils.logging import get_logger
class FlaxSuppressTokensLogitsProcessor(FlaxLogitsProcessor):
    """
    [`FlaxLogitsProcessor`] suppressing a list of tokens at each decoding step. The processor will set their log probs
    to be `-inf` so they are not sampled.

    Args:
        suppress_tokens (`list`):
            Tokens to not sample.
    """

    def __init__(self, suppress_tokens: list):
        self.suppress_tokens = list(suppress_tokens)

    def __call__(self, input_ids: jnp.ndarray, scores: jnp.ndarray, cur_len: int) -> jnp.ndarray:
        scores = scores.at[..., self.suppress_tokens].set(-float('inf'))
        return scores