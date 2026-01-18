import inspect
import jax
import jax.lax as lax
import jax.numpy as jnp
from ..utils import add_start_docstrings
from ..utils.logging import get_logger
class FlaxTopKLogitsWarper(FlaxLogitsWarper):
    """
    [`FlaxLogitsWarper`] that performs top-k, i.e. restricting to the k highest probability elements.

    Args:
        top_k (`int`):
            The number of highest probability vocabulary tokens to keep for top-k-filtering.
        filter_value (`float`, *optional*, defaults to -inf):
            All filtered values will be set to this float value.
        min_tokens_to_keep (`int`, *optional*, defaults to 1):
            Minimum number of tokens that cannot be filtered.
    """

    def __init__(self, top_k: int, filter_value: float=-float('Inf'), min_tokens_to_keep: int=1):
        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError(f'`top_k` has to be a strictly positive integer, but is {top_k}')
        self.top_k = max(top_k, min_tokens_to_keep)
        self.filter_value = filter_value

    def __call__(self, input_ids: jnp.ndarray, scores: jnp.ndarray, cur_len: int) -> jnp.ndarray:
        batch_size, vocab_size = scores.shape
        next_scores_flat = jnp.full(batch_size * vocab_size, self.filter_value)
        topk = min(self.top_k, scores.shape[-1])
        topk_scores, topk_indices = lax.top_k(scores, topk)
        shift = jnp.broadcast_to((jnp.arange(batch_size) * vocab_size)[:, None], (batch_size, topk)).flatten()
        topk_scores_flat = topk_scores.flatten()
        topk_indices_flat = topk_indices.flatten() + shift
        next_scores_flat = next_scores_flat.at[topk_indices_flat].set(topk_scores_flat)
        next_scores = next_scores_flat.reshape(batch_size, vocab_size)
        return next_scores