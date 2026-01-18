import re
import logging
import warnings
import json
from math import sqrt
import numpy as np
from .base import string_types
from .ndarray import NDArray, load
from . import random
from . import registry
from . import ndarray
from . util import is_np_array
from . import numpy as _mx_np  # pylint: disable=reimported
def _init_quantized_weight(self, _, arr):
    _arr = random.randint(-127, 127, dtype='int32').asnumpy()
    arr[:] = np.int8(_arr)