from typing import Tuple
import numpy as np
from ase.units import Bohr, Ha
from ase.data import covalent_radii
from ase.neighborlist import NeighborList
class BondPolarizability:

    def __init__(self, model=LippincottStuttman()):
        self.model = model

    def __call__(self, *args, **kwargs):
        """Shorthand for calculate"""
        return self.calculate(*args, **kwargs)

    def calculate(self, atoms, radiicut=1.5):
        """Sum up the bond polarizability from all bonds

        Parameters
        ----------
        atoms: Atoms object
        radiicut: float
          Bonds are counted up to
          radiicut * (sum of covalent radii of the pairs)
          Default: 1.5

        Returns
        -------
        polarizability tensor with unit (e^2 Angstrom^2 / eV).
        Multiply with Bohr * Ha to get (Angstrom^3)
        """
        radii = np.array([covalent_radii[z] for z in atoms.numbers])
        nl = NeighborList(radii * 1.5, skin=0, self_interaction=False)
        nl.update(atoms)
        pos_ac = atoms.get_positions()
        alpha = 0
        for ia, atom in enumerate(atoms):
            indices, offsets = nl.get_neighbors(ia)
            pos_ac = atoms.get_positions() - atoms.get_positions()[ia]
            for ib, offset in zip(indices, offsets):
                weight = 1
                if offset.any():
                    weight = 0.5
                dist_c = pos_ac[ib] + np.dot(offset, atoms.get_cell())
                dist = np.linalg.norm(dist_c)
                al, ap = self.model(atom.symbol, atoms[ib].symbol, dist)
                eye3 = np.eye(3) / 3
                alpha += weight * (al + 2 * ap) * eye3
                alpha += weight * (al - ap) * (np.outer(dist_c, dist_c) / dist ** 2 - eye3)
        return alpha / Bohr / Ha