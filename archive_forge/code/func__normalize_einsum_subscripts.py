import builtins
import collections
import functools
import math
import string
import warnings
import numpy as np
import tensorflow as tf
from tensorflow.experimental import numpy as tfnp
from tensorflow.python.ops.linalg.sparse import sparse_csr_matrix_ops
from keras.src.backend import config
from keras.src.backend import standardize_dtype
from keras.src.backend.common import dtypes
from keras.src.backend.common.backend_utils import canonicalize_axis
from keras.src.backend.common.backend_utils import to_tuple_or_list
from keras.src.backend.tensorflow import sparse
from keras.src.backend.tensorflow.core import cast
from keras.src.backend.tensorflow.core import convert_to_tensor
from keras.src.utils import tree
@functools.lru_cache(512)
def _normalize_einsum_subscripts(subscripts):
    mapping = {}
    normalized_subscripts = ''
    for c in subscripts:
        if c in string.ascii_letters:
            if c not in mapping:
                mapping[c] = string.ascii_letters[len(mapping)]
            normalized_subscripts += mapping[c]
        else:
            normalized_subscripts += c
    return normalized_subscripts