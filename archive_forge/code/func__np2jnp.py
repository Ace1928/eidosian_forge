import os
from typing import Dict, Optional, Union
import numpy as np
import jax.numpy as jnp
from jax import Array
from safetensors import numpy, safe_open
def _np2jnp(numpy_dict: Dict[str, np.ndarray]) -> Dict[str, Array]:
    for k, v in numpy_dict.items():
        numpy_dict[k] = jnp.array(v)
    return numpy_dict