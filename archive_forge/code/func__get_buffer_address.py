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
def _get_buffer_address(arr: np.ndarray) -> int:
    """Get the address of the buffer underlying the provided NumPy ndarray."""
    return arr.__array_interface__['data'][0]