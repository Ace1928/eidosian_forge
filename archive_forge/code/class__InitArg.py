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
class _InitArg:
    """Sentinel value returned by ``_init_arg()``."""