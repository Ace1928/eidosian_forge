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
def _init_bind(self, data_name: Optional[str]=None, identifiers: Optional[Tuple[str, ...]]=None):
    if self.bind.file.mode == 'r+':
        if 'type_id' not in self.info:
            self.info['type_id'] = self.type_id
        if 'data_name' not in self.info:
            self.info['data_name'] = data_name or self.__data_name__
        if 'identifiers' not in self.info:
            self.info['identifiers'] = identifiers or self.__identifiers__