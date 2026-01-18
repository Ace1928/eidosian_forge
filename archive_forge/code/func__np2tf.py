import os
from typing import Dict, Optional, Union
import numpy as np
import tensorflow as tf
from safetensors import numpy, safe_open
def _np2tf(numpy_dict: Dict[str, np.ndarray]) -> Dict[str, tf.Tensor]:
    for k, v in numpy_dict.items():
        numpy_dict[k] = tf.convert_to_tensor(v)
    return numpy_dict