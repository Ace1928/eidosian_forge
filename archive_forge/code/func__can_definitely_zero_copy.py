import ast
from collections.abc import Sequence
from concurrent import futures
import concurrent.futures.thread  # noqa
from copy import deepcopy
from itertools import zip_longest
import json
import operator
import re
import warnings
import numpy as np
import pyarrow as pa
from pyarrow.lib import _pandas_api, frombytes  # noqa
def _can_definitely_zero_copy(arr):
    return isinstance(arr, np.ndarray) and arr.flags.contiguous and issubclass(arr.dtype.type, np.integer)