from typing import Tuple, Type
from pennylane.data.base.attribute import DatasetAttribute
from pennylane.data.base.hdf5 import HDF5Array, HDF5Group
class DatasetString(DatasetAttribute[HDF5Array, str, str]):
    """Attribute type for strings."""
    type_id = 'string'

    @classmethod
    def consumes_types(cls) -> Tuple[Type[str]]:
        return (str,)

    def hdf5_to_value(self, bind: HDF5Array) -> str:
        return bind.asstr()[()]

    def value_to_hdf5(self, bind_parent: HDF5Group, key: str, value: str) -> HDF5Array:
        bind_parent[key] = value
        return bind_parent[key]