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
def __arrow_ext_scalar_class__(self):
    """
            ExtensionScalar subclass with custom logic for this array of tensors type.
            """
    return ArrowTensorScalar