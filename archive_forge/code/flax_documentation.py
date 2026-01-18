import os
from typing import Dict, Optional, Union
import numpy as np
import jax.numpy as jnp
from jax import Array
from safetensors import numpy, safe_open

    Loads a safetensors file into flax format.

    Args:
        filename (`str`, or `os.PathLike`)):
            The name of the file which contains the tensors

    Returns:
        `Dict[str, Array]`: dictionary that contains name as key, value as `Array`

    Example:

    ```python
    from safetensors.flax import load_file

    file_path = "./my_folder/bert.safetensors"
    loaded = load_file(file_path)
    ```
    