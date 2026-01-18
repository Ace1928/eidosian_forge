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
class _DatasetAttributeType(DatasetAttribute[HDF5Group, Dataset, Dataset]):
    """Attribute type for loading and saving datasets as attributes of
    datasets, or elements of collection types."""
    type_id = 'dataset'

    def hdf5_to_value(self, bind: HDF5Group) -> Dataset:
        return Dataset(bind)

    def value_to_hdf5(self, bind_parent: HDF5Group, key: str, value: Dataset) -> HDF5Group:
        hdf5.copy(value.bind, bind_parent, key)
        return bind_parent[key]