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
@PublicAPI(stability='beta')
class ArrowTensorScalar(pa.ExtensionScalar):

    def as_py(self) -> np.ndarray:
        return self.type._extension_scalar_to_ndarray(self)

    def __array__(self) -> np.ndarray:
        return self.as_py()