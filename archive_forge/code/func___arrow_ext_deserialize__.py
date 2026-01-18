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
@classmethod
def __arrow_ext_deserialize__(cls, storage_type, serialized):
    ndim = json.loads(serialized)
    dtype = storage_type['data'].type.value_type
    return cls(dtype, ndim)