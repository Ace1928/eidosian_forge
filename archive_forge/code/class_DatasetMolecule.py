from typing import Tuple, Type
from pennylane.data.base.attribute import DatasetAttribute
from pennylane.data.base.hdf5 import HDF5Group
from pennylane.data.base.mapper import AttributeTypeMapper
from pennylane.qchem import Molecule
class DatasetMolecule(DatasetAttribute[HDF5Group, Molecule, Molecule]):
    """Attribute type for ``pennylane.qchem.Molecule``."""
    type_id = 'molecule'

    @classmethod
    def consumes_types(cls) -> Tuple[Type[Molecule]]:
        return (Molecule,)

    def hdf5_to_value(self, bind: HDF5Group) -> Molecule:
        mapper = AttributeTypeMapper(bind)
        return Molecule(symbols=mapper['symbols'].copy_value(), coordinates=mapper['coordinates'].copy_value(), charge=mapper['charge'].copy_value(), mult=mapper['mult'].copy_value(), basis_name=mapper['basis_name'].copy_value(), l=mapper['l'].copy_value(), alpha=mapper['alpha'].copy_value(), coeff=mapper['coeff'].copy_value())

    def value_to_hdf5(self, bind_parent: HDF5Group, key: str, value: Molecule) -> HDF5Group:
        bind = bind_parent.create_group(key)
        mapper = AttributeTypeMapper(bind)
        mapper['symbols'] = value.symbols
        mapper['coordinates'] = value.coordinates
        mapper['charge'] = value.charge
        mapper['mult'] = value.mult
        mapper['basis_name'] = value.basis_name
        mapper['l'] = value.l
        mapper['alpha'] = value.alpha
        mapper['coeff'] = value.coeff
        return bind