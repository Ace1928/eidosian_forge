from __future__ import annotations
import itertools
from typing import TYPE_CHECKING
from monty.json import MSONable
from pymatgen.util.due import Doi, due
class MoleculeStructureComparator(MSONable):
    """
    Class to check whether the connection tables of the two molecules are the
    same. The atom in the two molecule must be paired accordingly.
    """
    ionic_element_list = ('Na', 'Mg', 'Al', 'Sc', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Rb', 'Sr')
    halogen_list = ('F', 'Cl', 'Br', 'I')

    def __init__(self, bond_length_cap=0.3, covalent_radius=CovalentRadius.radius, priority_bonds=(), priority_cap=0.8, ignore_ionic_bond=True, bond_13_cap=0.05):
        """
        Args:
            bond_length_cap: The ratio of the elongation of the bond to be
                acknowledged. If the distance between two atoms is less than (
                empirical covalent bond length) X (1 + bond_length_cap), the bond
                between the two atoms will be acknowledged.
            covalent_radius: The covalent radius of the atoms.
                dict (element symbol -> radius)
            priority_bonds: The bonds that are known to be existed in the initial
                molecule. Such bonds will be acknowledged in a loose criteria.
                The index should start from 0.
            priority_cap: The ratio of the elongation of the bond to be
                acknowledged for the priority bonds.
        """
        self.bond_length_cap = bond_length_cap
        self.covalent_radius = covalent_radius
        self.priority_bonds = [tuple(sorted(b)) for b in priority_bonds]
        self.priority_cap = priority_cap
        self.ignore_ionic_bond = ignore_ionic_bond
        self.ignore_halogen_self_bond = True
        self.bond_13_cap = bond_13_cap

    def are_equal(self, mol1, mol2) -> bool:
        """
        Compare the bond table of the two molecules.

        Args:
            mol1: first molecule. pymatgen Molecule object.
            mol2: second molecules. pymatgen Molecule object.
        """
        b1 = set(self._get_bonds(mol1))
        b2 = set(self._get_bonds(mol2))
        return b1 == b2

    @staticmethod
    def get_13_bonds(priority_bonds):
        """
        Args:
            priority_bonds ():

        Returns:
            tuple: 13 bonds
        """
        all_bond_pairs = list(itertools.combinations(priority_bonds, r=2))
        all_2_bond_atoms = [set(b1 + b2) for b1, b2 in all_bond_pairs]
        all_13_bond_atoms = [a for a in all_2_bond_atoms if len(a) == 3]
        all_2_and_13_bonds = {tuple(sorted(b)) for b in itertools.chain(*(itertools.combinations(p, 2) for p in all_13_bond_atoms))}
        bonds_13 = all_2_and_13_bonds - {tuple(b) for b in priority_bonds}
        return tuple(sorted(bonds_13))

    def _get_bonds(self, mol):
        """Find all bonds in a molecule.

        Args:
            mol: the molecule. pymatgen Molecule object

        Returns:
            List of tuple. Each tuple correspond to a bond represented by the
            id of the two end atoms.
        """
        n_atoms = len(mol)
        if self.ignore_ionic_bond:
            covalent_atoms = [idx for idx in range(n_atoms) if mol.species[idx].symbol not in self.ionic_element_list]
        else:
            covalent_atoms = list(range(n_atoms))
        all_pairs = list(itertools.combinations(covalent_atoms, 2))
        pair_dists = [mol.get_distance(*p) for p in all_pairs]
        unavailable_elements = set(mol.composition.as_dict()) - set(self.covalent_radius)
        if len(unavailable_elements) > 0:
            raise ValueError(f'The covalent radius for element {unavailable_elements} is not available')
        bond_13 = self.get_13_bonds(self.priority_bonds)
        max_length = [(self.covalent_radius[mol.sites[p[0]].specie.symbol] + self.covalent_radius[mol.sites[p[1]].specie.symbol]) * (1 + (self.priority_cap if p in self.priority_bonds else self.bond_length_cap if p not in bond_13 else self.bond_13_cap)) * (0.1 if self.ignore_halogen_self_bond and p not in self.priority_bonds and (mol.sites[p[0]].specie.symbol in self.halogen_list) and (mol.sites[p[1]].specie.symbol in self.halogen_list) else 1.0) for p in all_pairs]
        return [bond for bond, dist, cap in zip(all_pairs, pair_dists, max_length) if dist <= cap]

    def as_dict(self):
        """Returns: MSONable dict."""
        return {'version': __version__, '@module': type(self).__module__, '@class': type(self).__name__, 'bond_length_cap': self.bond_length_cap, 'covalent_radius': self.covalent_radius, 'priority_bonds': self.priority_bonds, 'priority_cap': self.priority_cap}

    @classmethod
    def from_dict(cls, dct: dict) -> Self:
        """
        Args:
            dct (dict): Dict representation.

        Returns:
            MoleculeStructureComparator
        """
        return cls(bond_length_cap=dct['bond_length_cap'], covalent_radius=dct['covalent_radius'], priority_bonds=dct['priority_bonds'], priority_cap=dct['priority_cap'])