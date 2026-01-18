from numbers import Number
from pennylane.data.base.attribute import DatasetAttribute
from pennylane.data.base.hdf5 import HDF5Array, HDF5Group
class DatasetScalar(DatasetAttribute[HDF5Array, Number, Number]):
    """
    Attribute type for numbers.
    """
    type_id = 'scalar'

    def hdf5_to_value(self, bind: HDF5Array) -> Number:
        return bind[()]

    def value_to_hdf5(self, bind_parent: HDF5Group, key: str, value: Number) -> HDF5Array:
        bind_parent[key] = value
        return bind_parent[key]