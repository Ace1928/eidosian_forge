from __future__ import annotations
import gzip
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any
import numpy as np
from pymatgen.core import Lattice, Molecule, Structure
from pymatgen.core.tensors import Tensor
def _parse_structure(self) -> Structure | Molecule:
    """Parse a structure object from the file.

        For the given section of the aims output file generate the
        calculated structure.

        Returns:
            The structure or molecule for the calculation
        """
    species, coords, velocities, lattice = self._parse_lattice_atom_pos()
    site_properties: dict[str, Sequence[Any]] = dict()
    if len(velocities) > 0:
        site_properties['velocity'] = np.array(velocities)
    results = self.results
    site_prop_keys = {'forces': 'force', 'stresses': 'atomic_virial_stress', 'hirshfeld_charges': 'hirshfeld_charge', 'hirshfeld_volumes': 'hirshfeld_volume', 'hirshfeld_atomic_dipoles': 'hirshfeld_atomic_dipole'}
    properties = {prop: results[prop] for prop in results if prop not in site_prop_keys}
    for prop, site_key in site_prop_keys.items():
        if prop in results:
            site_properties[site_key] = results[prop]
    if lattice is not None:
        return Structure(lattice, species, coords, site_properties=site_properties, properties=properties, coords_are_cartesian=True)
    return Molecule(species, coords, site_properties=site_properties, properties=properties)