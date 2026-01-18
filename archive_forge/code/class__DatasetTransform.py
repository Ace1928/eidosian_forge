import typing
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import (
from typing_extensions import dataclass_transform  # pylint: disable=no-name-in-module
from pennylane.data.base import hdf5
from pennylane.data.base.attribute import AttributeInfo, DatasetAttribute
from pennylane.data.base.hdf5 import HDF5Any, HDF5Group, h5py
from pennylane.data.base.mapper import MapperMixin, match_obj_type
from pennylane.data.base.typing_util import UNSET, T
@dataclass_transform(order_default=False, eq_default=False, kw_only_default=True, field_specifiers=(field, _init_arg))
class _DatasetTransform:
    """This base class that tells the type system that ``Dataset`` behaves like a dataclass.
    See: https://peps.python.org/pep-0681/
    """