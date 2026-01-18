import itertools
import json
import sys
from typing import Iterable, Optional, Tuple, List, Sequence, Union
from pkg_resources._vendor.packaging.version import parse as parse_version
import numpy as np
import pyarrow as pa
from ray.air.util.tensor_extensions.utils import (
from ray._private.utils import _get_pyarrow_version
from ray.util.annotations import PublicAPI
def _extension_scalar_to_ndarray(self, scalar: pa.ExtensionScalar) -> np.ndarray:
    """
            Convert an ExtensionScalar to a tensor element.
            """
    data = scalar.value.get('data')
    raw_values = data.values
    shape = tuple(scalar.value.get('shape').as_py())
    value_type = raw_values.type
    offset = raw_values.offset
    data_buffer = raw_values.buffers()[1]
    return _to_ndarray_helper(shape, value_type, offset, data_buffer)