import inspect
import jax
import jax.lax as lax
import jax.numpy as jnp
from ..utils import add_start_docstrings
from ..utils.logging import get_logger
class FlaxLogitsWarper:
    """Abstract base class for all logit warpers that can be applied during generation with multinomial sampling."""

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: jnp.ndarray, scores: jnp.ndarray) -> jnp.ndarray:
        """Flax method for warping logits."""
        raise NotImplementedError(f'{self.__class__} is an abstract class. Only classes inheriting this class can be called.')