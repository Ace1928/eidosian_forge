import inspect
import jax
import jax.lax as lax
import jax.numpy as jnp
from ..utils import add_start_docstrings
from ..utils.logging import get_logger
def _force_token(generation_idx):
    batch_size = scores.shape[0]
    current_token = self.force_token_array[generation_idx]
    new_scores = jnp.ones_like(scores, dtype=scores.dtype) * -float('inf')
    updates = jnp.zeros((batch_size, 1), dtype=scores.dtype)
    new_scores = lax.dynamic_update_slice(new_scores, updates, (0, current_token))
    return new_scores