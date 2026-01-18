import typing
from collections.abc import MutableMapping
from types import MappingProxyType
from typing import Any, Dict, Optional, Type
from pennylane.data.base.attribute import (
from pennylane.data.base.hdf5 import HDF5Any, HDF5Group
class DatasetNotWriteableError(RuntimeError):
    """Exception raised when attempting to set an attribute
    on a dataset whose underlying file is not writeable."""

    def __init__(self, bind: HDF5Any):
        self.bind = bind
        super().__init__(f'Dataset file is not writeable: {bind.filename}')