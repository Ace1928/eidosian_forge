from __future__ import annotations
import gzip
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any
import numpy as np
from pymatgen.core import Lattice, Molecule, Structure
from pymatgen.core.tensors import Tensor
def _parse_lattice_atom_pos(self) -> tuple[list[str], list[Vector3D], list[Vector3D], Lattice | None]:
    """Parse the lattice and atomic positions of the structure

        Returns:
            list[str]: The species symbols for the atoms in the structure
            list[Vector3D]: The Cartesian coordinates of the atoms
            list[Vector3D]: The velocities of the atoms
            Lattice or None: The Lattice for the structure
        """
    lattice_vectors = []
    velocities: list[Vector3D] = []
    species: list[str] = []
    coords: list[Vector3D] = []
    start_keys = ['Atomic structure (and velocities) as used in the preceding time step', 'Updated atomic structure', 'Atomic structure that was used in the preceding time step of the wrapper']
    line_start = self.reverse_search_for(start_keys)
    if line_start == LINE_NOT_FOUND:
        species = [sp.symbol for sp in self.initial_structure.species]
        coords = self.initial_structure.cart_coords.tolist()
        velocities = list(self.initial_structure.site_properties.get('velocity', []))
        lattice = self.initial_lattice
        return (species, coords, velocities, lattice)
    line_start += 1
    line_end = self.reverse_search_for(['Writing the current geometry to file "geometry.in.next_step"'], line_start)
    if line_end == LINE_NOT_FOUND:
        line_end = len(self.lines)
    for line in self.lines[line_start:line_end]:
        if 'lattice_vector   ' in line:
            lattice_vectors.append([float(inp) for inp in line.split()[1:]])
        elif 'atom   ' in line:
            line_split = line.split()
            species.append(line_split[4])
            coords.append([float(inp) for inp in line_split[1:4]])
        elif 'velocity   ' in line:
            velocities.append([float(inp) for inp in line.split()[1:]])
    lattice = Lattice(lattice_vectors) if len(lattice_vectors) == 3 else None
    return (species, coords, velocities, lattice)