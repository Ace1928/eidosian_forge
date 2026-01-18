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
class _ArrowTensorScalarIndexingMixin:
    """
    A mixin providing support for scalar indexing in tensor extension arrays for
    Arrow < 9.0.0, before full ExtensionScalar support was added. This mixin overrides
    __getitem__, __iter__, and to_pylist.
    """
    if not _arrow_extension_scalars_are_subclassable():

        def __iter__(self):
            for i in range(len(self)):
                yield self.__getitem__(i)

        def to_pylist(self):
            return list(self)
        if _arrow_supports_extension_scalars():

            def __getitem__(self, key):
                item = super().__getitem__(key)
                if not isinstance(key, slice):
                    item = item.type._extension_scalar_to_ndarray(item)
                return item
        else:

            def __getitem__(self, key):
                if isinstance(key, slice):
                    return super().__getitem__(key)
                return self._to_numpy(key)