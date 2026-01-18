import inspect
import jax
import jax.lax as lax
import jax.numpy as jnp
from ..utils import add_start_docstrings
from ..utils.logging import get_logger
class FlaxForcedEOSTokenLogitsProcessor(FlaxLogitsProcessor):
    """
    [`FlaxLogitsProcessor`] that enforces the specified token as the last generated token when `max_length` is reached.

    Args:
        max_length (`int`):
            The maximum length of the sequence to be generated.
        eos_token_id (`int`):
            The id of the token to force as the last generated token when `max_length` is reached.
    """

    def __init__(self, max_length: int, eos_token_id: int):
        self.max_length = max_length
        self.eos_token_id = eos_token_id

    def __call__(self, input_ids: jnp.ndarray, scores: jnp.ndarray, cur_len: int) -> jnp.ndarray:
        new_scores = jnp.full(scores.shape, -float('inf'))
        apply_penalty = 1 - jnp.bool_(cur_len - self.max_length + 1)
        scores = jnp.where(apply_penalty, new_scores.at[:, self.eos_token_id].set(0), scores)
        return scores