import inspect
import jax
import jax.lax as lax
import jax.numpy as jnp
from ..utils import add_start_docstrings
from ..utils.logging import get_logger
class FlaxTemperatureLogitsWarper(FlaxLogitsWarper):
    """
    [`FlaxLogitsWarper`] for temperature (exponential scaling output probability distribution).

    Args:
        temperature (`float`):
            The value used to module the logits distribution.
    """

    def __init__(self, temperature: float):
        if not isinstance(temperature, float) or not temperature > 0:
            raise ValueError(f'`temperature` has to be a strictly positive float, but is {temperature}')
        self.temperature = temperature

    def __call__(self, input_ids: jnp.ndarray, scores: jnp.ndarray, cur_len: int) -> jnp.ndarray:
        scores = scores / self.temperature
        return scores