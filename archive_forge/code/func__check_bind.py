import typing
import warnings
from abc import ABC, abstractmethod
from collections.abc import Mapping, MutableMapping, Sequence
from functools import lru_cache
from numbers import Number
from types import MappingProxyType
from typing import (
from pennylane.data.base import hdf5
from pennylane.data.base.hdf5 import HDF5, HDF5Any, HDF5Group
from pennylane.data.base.typing_util import UNSET, get_type, get_type_str
def _check_bind(self):
    """
        Checks that ``bind.attrs`` contains the type_id corresponding to
        this type.
        """
    existing_type_id = self.info.get('type_id')
    if existing_type_id is None:
        raise ValueError("'bind' does not contain a dataset attribute.")
    if existing_type_id != self.type_id:
        raise TypeError(f"'bind' is bound to another attribute type '{existing_type_id}'")