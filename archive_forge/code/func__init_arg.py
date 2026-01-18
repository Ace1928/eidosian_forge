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
def _init_arg(default: Any, alias: Optional[str]=None, kw_only: bool=False) -> Any:
    """This function exists only for the benefit of the type checker. It is used to
    annotate attributes on ``Dataset`` that are not part of the data model, but
    should appear in the generated ``__init__`` method.
    """
    return _InitArg