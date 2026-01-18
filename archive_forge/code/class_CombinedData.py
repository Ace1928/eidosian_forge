from __future__ import annotations
import itertools
import re
import warnings
from io import StringIO
from pathlib import Path
from typing import TYPE_CHECKING, Literal
import numpy as np
import pandas as pd
from monty.io import zopen
from monty.json import MSONable
from monty.serialization import loadfn
from ruamel.yaml import YAML
from pymatgen.core import Element, Lattice, Molecule, Structure
from pymatgen.core.operations import SymmOp
from pymatgen.util.io_utils import clean_lines
class CombinedData(LammpsData):
    """
    Object for a collective set of data for a series of LAMMPS data file.
    velocities not yet implemented.
    """

    def __init__(self, list_of_molecules: list, list_of_names: list[str], list_of_numbers: list[int], coordinates: pd.DataFrame, atom_style: str='full') -> None:
        """
        Args:
            list_of_molecules: A list of LammpsData objects of a chemical cluster.
                Each LammpsData object (cluster) may contain one or more molecule ID.
            list_of_names: A list of name (string) for each cluster. The characters in each name are
                restricted to word characters ([a-zA-Z0-9_]). If names with any non-word characters
                are passed in, the special characters will be substituted by '_'.
            list_of_numbers: A list of Integer for counts of each molecule
            coordinates (pandas.DataFrame): DataFrame at least containing
                columns of ["x", "y", "z"] for coordinates of atoms.
            atom_style (str): Output atom_style. Default to "full".
        """
        self._list_of_molecules = list_of_molecules
        self._list_of_names = list_of_names
        self._list_of_numbers = list_of_numbers
        self._coordinates = coordinates
        self._coordinates.index = self._coordinates.index.map(int)
        max_xyz = self._coordinates[['x', 'y', 'z']].max().max()
        min_xyz = self._coordinates[['x', 'y', 'z']].min().min()
        self.box = LammpsBox(np.array(3 * [[min_xyz - 0.5, max_xyz + 0.5]]))
        self.atom_style = atom_style
        self.n = sum(self._list_of_numbers)
        self.names = []
        for name in self._list_of_names:
            self.names.append('_'.join(re.findall('\\w+', name)))
        self.mols = self._list_of_molecules
        self.nums = self._list_of_numbers
        self.masses = pd.concat([mol.masses.copy() for mol in self.mols], ignore_index=True)
        self.masses.index += 1
        all_ff_kws = SECTION_KEYWORDS['ff'] + SECTION_KEYWORDS['class2']
        appeared_kws = {k for mol in self.mols if mol.force_field is not None for k in mol.force_field}
        ff_kws = [k for k in all_ff_kws if k in appeared_kws]
        self.force_field = {}
        for kw in ff_kws:
            self.force_field[kw] = pd.concat([mol.force_field[kw].copy() for mol in self.mols if kw in (mol.force_field or [])], ignore_index=True)
            self.force_field[kw].index += 1
        if not bool(self.force_field):
            self.force_field = None
        self.atoms = pd.DataFrame()
        mol_count = 0
        type_count = 0
        self.mols_per_data = []
        for idx, mol in enumerate(self.mols):
            atoms_df = mol.atoms.copy()
            atoms_df['molecule-ID'] += mol_count
            atoms_df['type'] += type_count
            mols_in_data = len(atoms_df['molecule-ID'].unique())
            self.mols_per_data.append(mols_in_data)
            for _ in range(self.nums[idx]):
                self.atoms = pd.concat([self.atoms, atoms_df], ignore_index=True)
                atoms_df['molecule-ID'] += mols_in_data
            type_count += len(mol.masses)
            mol_count += self.nums[idx] * mols_in_data
        self.atoms.index += 1
        assert len(self.atoms) == len(self._coordinates), 'Wrong number of coordinates'
        self.atoms.update(self._coordinates)
        self.velocities = None
        assert self.mols[0].velocities is None, 'Velocities not supported'
        self.topology = {}
        atom_count = 0
        count = {'Bonds': 0, 'Angles': 0, 'Dihedrals': 0, 'Impropers': 0}
        for idx, mol in enumerate(self.mols):
            for kw in SECTION_KEYWORDS['topology']:
                if mol.topology and kw in mol.topology:
                    if kw not in self.topology:
                        self.topology[kw] = pd.DataFrame()
                    topo_df = mol.topology[kw].copy()
                    topo_df['type'] += count[kw]
                    for col in topo_df.columns[1:]:
                        topo_df[col] += atom_count
                    for _ in range(self.nums[idx]):
                        self.topology[kw] = pd.concat([self.topology[kw], topo_df], ignore_index=True)
                        for col in topo_df.columns[1:]:
                            topo_df[col] += len(mol.atoms)
                    count[kw] += len(mol.force_field[kw[:-1] + ' Coeffs'])
            atom_count += len(mol.atoms) * self.nums[idx]
        for kw in SECTION_KEYWORDS['topology']:
            if kw in self.topology:
                self.topology[kw].index += 1
        if not self.topology:
            self.topology = None

    @property
    def structure(self) -> Structure:
        """
        Exports a periodic structure object representing the simulation
        box.

        Returns:
            Structure
        """
        ld_cp = self.as_lammpsdata()
        return ld_cp.structure

    def disassemble(self, atom_labels: Sequence[str] | None=None, guess_element: bool=True, ff_label: str='ff_map'):
        """
        Breaks down each LammpsData in CombinedData to building blocks
        (LammpsBox, ForceField and a series of Topology).
        RESTRICTIONS APPLIED:
        1. No complex force field defined not just on atom
            types, where the same type or equivalent types of topology
            may have more than one set of coefficients.
        2. No intermolecular topologies (with atoms from different
            molecule-ID) since a Topology object includes data for ONE
            molecule or structure only.

        Args:
            atom_labels ([str]): List of strings (must be different
                from one another) for labelling each atom type found in
                Masses section. Default to None, where the labels are
                automatically added based on either element guess or
                dummy specie assignment.
            guess_element (bool): Whether to guess the element based on
                its atomic mass. Default to True, otherwise dummy
                species "Qa", "Qb", ... will be assigned to various
                atom types. The guessed or assigned elements will be
                reflected on atom labels if atom_labels is None, as
                well as on the species of molecule in each Topology.
            ff_label (str): Site property key for labeling atoms of
                different types. Default to "ff_map".

        Returns:
            [(LammpsBox, ForceField, [Topology]), ...]
        """
        return [mol.disassemble(atom_labels=atom_labels, guess_element=guess_element, ff_label=ff_label) for mol in self.mols]

    @classmethod
    def from_ff_and_topologies(cls) -> None:
        """Unsupported constructor for CombinedData objects."""
        raise AttributeError('Unsupported constructor for CombinedData objects')

    @classmethod
    def from_structure(cls) -> None:
        """Unsupported constructor for CombinedData objects."""
        raise AttributeError('Unsupported constructor for CombinedData objects')

    @classmethod
    def parse_xyz(cls, filename: str | Path) -> pd.DataFrame:
        """
        Load xyz file generated from packmol (for those who find it hard to install openbabel).

        Returns:
            pandas.DataFrame
        """
        with zopen(filename, mode='rt') as file:
            lines = file.readlines()
        sio = StringIO(''.join(lines[2:]))
        df = pd.read_csv(sio, header=None, comment='#', delim_whitespace=True, names=['atom', 'x', 'y', 'z'])
        df.index += 1
        return df

    @classmethod
    def from_files(cls, coordinate_file: str, list_of_numbers: list[int], *filenames: str) -> Self:
        """
        Constructor that parse a series of data file.

        Args:
            coordinate_file (str): The filename of xyz coordinates.
            list_of_numbers (list[int]): A list of numbers specifying counts for each
                clusters parsed from files.
            filenames (str): A series of LAMMPS data filenames in string format.
        """
        names = []
        mols = []
        styles = []
        clusters = []
        for idx, filename in enumerate(filenames, start=1):
            cluster = LammpsData.from_file(filename)
            clusters.append(cluster)
            names.append(f'cluster{idx}')
            mols.append(cluster)
            styles.append(cluster.atom_style)
        if len(set(styles)) != 1:
            raise ValueError('Files have different atom styles.')
        coordinates = cls.parse_xyz(filename=coordinate_file)
        return cls.from_lammpsdata(mols, names, list_of_numbers, coordinates, styles.pop())

    @classmethod
    def from_lammpsdata(cls, mols: list, names: list, list_of_numbers: list, coordinates: pd.DataFrame, atom_style: str | None=None) -> Self:
        """
        Constructor that can infer atom_style.
        The input LammpsData objects are used non-destructively.

        Args:
            mols: a list of LammpsData of a chemical cluster.Each LammpsData object (cluster)
                may contain one or more molecule ID.
            names: a list of name for each cluster.
            list_of_numbers: a list of Integer for counts of each molecule
            coordinates (pandas.DataFrame): DataFrame at least containing
                columns of ["x", "y", "z"] for coordinates of atoms.
            atom_style (str): Output atom_style. Default to "full".
        """
        styles = [mol.atom_style for mol in mols]
        if len(set(styles)) != 1:
            raise ValueError('Data have different atom_style.')
        style_return = styles.pop()
        if atom_style and atom_style != style_return:
            raise ValueError('Data have different atom_style as specified.')
        return cls(mols, names, list_of_numbers, coordinates, style_return)

    def get_str(self, distance: int=6, velocity: int=8, charge: int=4, hybrid: bool=True) -> str:
        """
        Returns the string representation of CombinedData, essentially
        the string to be written to a file. Combination info is included
        as a comment. For single molecule ID data, the info format is:
            num name
        For data with multiple molecule ID, the format is:
            num(mols_per_data) name.

        Args:
            distance (int): No. of significant figures to output for
                box settings (bounds and tilt) and atomic coordinates.
                Default to 6.
            velocity (int): No. of significant figures to output for
                velocities. Default to 8.
            charge (int): No. of significant figures to output for
                charges. Default to 4.
            hybrid (bool): Whether to write hybrid coeffs types.
                Default to True. If the data object has no hybrid
                coeffs types and has large coeffs section, one may
                use False to speed up the process. Otherwise, the
                default is recommended.

        Returns:
            String representation
        """
        lines = LammpsData.get_str(self, distance, velocity, charge, hybrid).splitlines()
        info = '# ' + ' + '.join((f'{a} {b}' if c == 1 else f'{a}({c}) {b}' for a, b, c in zip(self.nums, self.names, self.mols_per_data)))
        lines.insert(1, info)
        return '\n'.join(lines)

    def as_lammpsdata(self):
        """
        Convert a CombinedData object to a LammpsData object. attributes are deep-copied.

        box (LammpsBox): Simulation box.
        force_fieldct (dict): Data for force field sections. Optional
            with default to None. Only keywords in force field and
            class 2 force field are valid keys, and each value is a
            DataFrame.
        topology (dict): Data for topology sections. Optional with
            default to None. Only keywords in topology are valid
            keys, and each value is a DataFrame.
        """
        items = {}
        items['box'] = LammpsBox(self.box.bounds, self.box.tilt)
        items['masses'] = self.masses.copy()
        items['atoms'] = self.atoms.copy()
        items['atom_style'] = self.atom_style
        items['velocities'] = None
        if self.force_field:
            all_ff_kws = SECTION_KEYWORDS['ff'] + SECTION_KEYWORDS['class2']
            items['force_field'] = {k: v.copy() for k, v in self.force_field.items() if k in all_ff_kws}
        if self.topology:
            items['topology'] = {k: v.copy() for k, v in self.topology.items() if k in SECTION_KEYWORDS['topology']}
        return LammpsData(**items)