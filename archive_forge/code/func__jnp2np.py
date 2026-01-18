import os
from typing import Dict, Optional, Union
import numpy as np
import jax.numpy as jnp
from jax import Array
from safetensors import numpy, safe_open
def _jnp2np(jnp_dict: Dict[str, Array]) -> Dict[str, np.array]:
    for k, v in jnp_dict.items():
        jnp_dict[k] = np.asarray(v)
    return jnp_dict