from __future__ import annotations
import re
import warnings
from typing import TYPE_CHECKING
import numpy as np
from monty.io import zopen
from monty.json import MSONable
from tabulate import tabulate
from pymatgen.core import Element, Lattice, Molecule, Structure
from pymatgen.io.cif import CifParser
from pymatgen.io.core import ParseError
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.util.io_utils import clean_lines
from pymatgen.util.string import str_delimited
class Atoms(MSONable):
    """Atomic cluster centered around the absorbing atom."""

    def __init__(self, struct, absorbing_atom, radius):
        """
        Args:
            struct (Structure): input structure
            absorbing_atom (str/int): Symbol for absorbing atom or site index
            radius (float): radius of the atom cluster in Angstroms.
        """
        if not struct.is_ordered:
            raise ValueError('Structure with partial occupancies cannot be converted into atomic coordinates!')
        self.struct = struct
        self.absorbing_atom, self.center_index = get_absorbing_atom_symbol_index(absorbing_atom, struct)
        self.radius = radius
        self._cluster = self._set_cluster()
        self.pot_dict = get_atom_map(self._cluster, self.absorbing_atom)

    def _set_cluster(self):
        """
        Compute and set the cluster of atoms as a Molecule object. The site
        coordinates are translated such that the absorbing atom (aka central
        atom) is at the origin.

        Returns:
            Molecule
        """
        center = self.struct[self.center_index].coords
        sphere = self.struct.get_neighbors(self.struct[self.center_index], self.radius)
        symbols = [self.absorbing_atom]
        coords = [[0, 0, 0]]
        for site_dist in sphere:
            site_symbol = re.sub('[^aA-zZ]+', '', site_dist[0].species_string)
            symbols.append(site_symbol)
            coords.append(site_dist[0].coords - center)
        return Molecule(symbols, coords)

    @property
    def cluster(self):
        """Returns the atomic cluster as a Molecule object."""
        return self._cluster

    @staticmethod
    def atoms_string_from_file(filename):
        """
        Reads atomic shells from file such as feff.inp or ATOMS file
        The lines are arranged as follows:

        x y z   ipot    Atom Symbol   Distance   Number

        with distance being the shell radius and ipot an integer identifying
        the potential used.

        Args:
            filename: File name containing atomic coord data.

        Returns:
            Atoms string.
        """
        with zopen(filename, mode='rt') as fobject:
            f = fobject.readlines()
            coords = 0
            atoms_str = []
            for line in f:
                if coords == 0:
                    find_atoms = line.find('ATOMS')
                    if find_atoms >= 0:
                        coords = 1
                if coords == 1 and 'END' not in line:
                    atoms_str.append(line.replace('\r', ''))
        return ''.join(atoms_str)

    @staticmethod
    def cluster_from_file(filename):
        """
        Parse the feff input file and return the atomic cluster as a Molecule
        object.

        Args:
            filename (str): path the feff input file

        Returns:
            Molecule: the atomic cluster as Molecule object. The absorbing atom
                is the one at the origin.
        """
        atoms_string = Atoms.atoms_string_from_file(filename)
        lines = [line.split() for line in atoms_string.splitlines()[1:]]
        coords = []
        symbols = []
        for tokens in lines:
            if tokens and (not tokens[0].startswith('*')):
                coords.append([float(val) for val in tokens[:3]])
                symbols.append(tokens[4])
        return Molecule(symbols, coords)

    def get_lines(self) -> list[list[str | int]]:
        """
        Returns a list of string representations of the atomic configuration
        information(x, y, z, ipot, atom_symbol, distance, id).

        Returns:
            list[list[str | int]]: lines sorted by the distance from the absorbing atom.
        """
        lines = [[f'{self._cluster[0].x}', f'{self._cluster[0].y}', f'{self._cluster[0].z}', 0, self.absorbing_atom, '0.0', 0]]
        for idx, site in enumerate(self._cluster[1:], start=1):
            site_symbol = site.specie.symbol
            ipot = self.pot_dict[site_symbol]
            dist = self._cluster.get_distance(0, idx)
            lines += [[f'{site.x}', f'{site.y}', f'{site.z}', ipot, site_symbol, f'{dist}', idx]]
        return sorted(lines, key=lambda line: float(line[5]))

    def __str__(self):
        """String representation of Atoms file."""
        lines_sorted = self.get_lines()
        lines_formatted = str(tabulate(lines_sorted, headers=['*       x', 'y', 'z', 'ipot', 'Atom', 'Distance', 'Number']))
        atom_list = lines_formatted.replace('--', '**')
        return f'ATOMS\n{atom_list}\nEND\n'

    def write_file(self, filename='ATOMS'):
        """
        Write Atoms list to file.

        Args:
            filename: path for file to be written
        """
        with zopen(filename, mode='wt') as file:
            file.write(f'{self}\n')