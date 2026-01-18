from __future__ import annotations
import linecache
from abc import ABC, abstractmethod
from collections import Counter
from typing import TYPE_CHECKING
import numpy as np
from monty.io import zopen
from monty.json import MSONable
from pymatgen.core import Lattice, Structure
from pymatgen.symmetry.kpath import KPathSeek
class ACstrExtractor(ACExtractorBase):
    """Extract information from atom.config file. You can get str by slicing the MOVEMENT."""

    def __init__(self, atom_config_str: str):
        """Initialization function.

        Args:
            atom_config_str (str): A string describing the structure in atom.config file.
        """
        self.atom_config_str = atom_config_str
        self.strs_lst = self.atom_config_str.split('\n')
        self.num_atoms = self.get_n_atoms()

    def get_n_atoms(self) -> int:
        """Return the number of atoms in structure.

        Returns:
            int: The number of atoms
        """
        return int(self.strs_lst[0].split()[0].strip())

    def get_lattice(self) -> np.ndarray:
        """Return the lattice of structure.

        Returns:
            np.ndarray: Lattice basis vectors of shape=(9,)
        """
        basis_vectors_lst = []
        aim_content = 'LATTICE'
        aim_idx = ListLocator.locate_all_lines(strs_lst=self.strs_lst, content=aim_content)[0]
        for idx_str in [aim_idx + 1, aim_idx + 2, aim_idx + 3]:
            str_lst = self.strs_lst[idx_str].split()[:3]
            for tmp_str in str_lst:
                basis_vectors_lst.append(float(tmp_str))
        return np.array(basis_vectors_lst)

    def get_types(self) -> np.ndarray:
        """Return the atomic number of atoms in structure.

        Returns:
            np.ndarray: Types of elements.
        """
        aim_content = 'POSITION'
        aim_idx = ListLocator.locate_all_lines(strs_lst=self.strs_lst, content=aim_content)[0]
        strs_lst = self.strs_lst[aim_idx + 1:aim_idx + self.num_atoms + 1]
        atomic_numbers_lst = [int(entry.split()[0]) for entry in strs_lst]
        return np.array(atomic_numbers_lst)

    def get_coords(self) -> np.ndarray:
        """Return the fractional coordinate of atoms in structure.

        Returns:
            np.ndarray: Fractional coordinates of atoms of shape=(num_atoms*3,)
        """
        coords_lst = []
        aim_content = 'POSITION'
        aim_idx = ListLocator.locate_all_lines(strs_lst=self.strs_lst, content=aim_content)[0]
        for tmp_str in self.strs_lst[aim_idx + 1:aim_idx + self.num_atoms + 1]:
            tmp_strs_lst = tmp_str.split()
            tmp_coord = [float(value) for value in tmp_strs_lst[1:4]]
            coords_lst.append(tmp_coord)
        return np.array(coords_lst).reshape(-1)

    def get_magmoms(self) -> np.ndarray:
        """Return the magnetic moments of atoms in structure.

        Returns:
            np.ndarray: Atomic magnetic moments.
        """
        magnetic_moments_lst: list[float] = []
        aim_content: str = 'MAGNETIC'
        aim_idxs: list[int] = ListLocator.locate_all_lines(strs_lst=self.strs_lst, content=aim_content)
        if len(aim_idxs) == 0:
            magnetic_moments_lst = [0.0 for _ in range(self.num_atoms)]
        else:
            aim_idx = aim_idxs[0]
            magnetic_moments_content = self.strs_lst[aim_idx + 1:aim_idx + self.num_atoms + 1]
            magnetic_moments_lst = [float(tmp_magnetic_moment.split()[-1]) for tmp_magnetic_moment in magnetic_moments_content]
        return np.array(magnetic_moments_lst)

    def get_e_tot(self) -> np.ndarray:
        """Return the total energy of structure.

        Returns:
            np.ndarray: The total energy of the material system.
        """
        strs_lst = self.strs_lst[0].split(',')
        aim_index = ListLocator.locate_all_lines(strs_lst=strs_lst, content='EK (EV) =')[0]
        return np.array([float(strs_lst[aim_index].split()[3].strip())])

    def get_atom_energies(self) -> np.ndarray | None:
        """Return the energies of individual atoms in material system.

        Returns:
            np.ndarray | None : The energies of individual atoms within the material system.

        Description:
            When turn on `ENERGY DEPOSITION`, PWmat will output energy per atom.
        """
        energies = []
        aim_content = 'Atomic-Energy, '.upper()
        aim_idxs = ListLocator.locate_all_lines(strs_lst=self.strs_lst, content=aim_content)
        if len(aim_idxs) == 0:
            return None
        aim_idx = aim_idxs[0]
        for tmp_str in self.strs_lst[aim_idx + 1:aim_idx + self.num_atoms + 1]:
            '\n            Atomic-Energy, Etot(eV),E_nonloc(eV),Q_atom:dE(eV)=  -0.1281163115E+06\n            14   0.6022241483E+03    0.2413350871E+02    0.3710442365E+01\n            '
            energies.append(float(tmp_str.split()[1]))
        return np.array(energies)

    def get_atom_forces(self) -> np.ndarray:
        """Return the force on atoms in material system.

        Returns:
            np.ndarray: Forces acting on individual atoms of shape=(num_atoms*3,)
        """
        forces = []
        aim_content = 'Force'.upper()
        aim_idx = ListLocator.locate_all_lines(strs_lst=self.strs_lst, content=aim_content)[0]
        for line in self.strs_lst[aim_idx + 1:aim_idx + self.num_atoms + 1]:
            forces.append([float(val) for val in line.split()[1:4]])
        return -np.array(forces).reshape(-1)

    def get_virial(self) -> np.ndarray | None:
        """Return the virial tensor of material system.

        Returns:
            np.ndarray | None: Virial tensor of shape=(9,)
        """
        virial_tensor = []
        aim_content = 'LATTICE'
        aim_idx = ListLocator.locate_all_lines(strs_lst=self.strs_lst, content=aim_content)[0]
        for tmp_idx in [aim_idx + 1, aim_idx + 2, aim_idx + 3]:
            tmp_strs_lst = self.strs_lst[tmp_idx].split()
            tmp_aim_row_lst = ListLocator.locate_all_lines(strs_lst=tmp_strs_lst, content='STRESS')
            if len(tmp_aim_row_lst) == 0:
                return None
        for tmp_idx in [aim_idx + 1, aim_idx + 2, aim_idx + 3]:
            tmp_str_lst = self.strs_lst[tmp_idx].split()[-3:]
            virial_tensor.append(float(tmp_str_lst[0]))
            virial_tensor.append(float(tmp_str_lst[1]))
            virial_tensor.append(float(tmp_str_lst[2]))
        return np.array(virial_tensor)