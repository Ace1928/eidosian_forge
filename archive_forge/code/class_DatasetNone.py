from typing import Literal, Tuple, Type
from pennylane.data.base.attribute import DatasetAttribute
from pennylane.data.base.hdf5 import HDF5Array, HDF5Group
class DatasetNone(DatasetAttribute[HDF5Array, type(None), type(None)]):
    """Datasets type for 'None' values."""
    type_id = 'none'

    @classmethod
    def default_value(cls) -> Literal[None]:
        return None

    @classmethod
    def consumes_types(cls) -> Tuple[Type[None]]:
        return (type(None),)

    def hdf5_to_value(self, bind) -> None:
        """Returns None."""
        return None

    def value_to_hdf5(self, bind_parent: HDF5Group, key: str, value: None) -> HDF5Array:
        """Creates an empty HDF5 array under 'key'."""
        return bind_parent.create_dataset(key, dtype='f')

    def __bool__(self) -> Literal[False]:
        return False