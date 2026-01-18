from math import sqrt
from warnings import warn
import numpy as np
from scipy.linalg import expm, logm
from ase.calculators.calculator import PropertyNotImplementedError
from ase.geometry import (find_mic, wrap_positions, get_distances_derivatives,
from ase.utils.parsemath import eval_expression
from ase.stress import (full_3x3_to_voigt_6_stress,
class FixAtoms(FixConstraint):
    """Constraint object for fixing some chosen atoms."""

    def __init__(self, indices=None, mask=None):
        """Constrain chosen atoms.

        Parameters
        ----------
        indices : list of int
           Indices for those atoms that should be constrained.
        mask : list of bool
           One boolean per atom indicating if the atom should be
           constrained or not.

        Examples
        --------
        Fix all Copper atoms:

        >>> mask = [s == 'Cu' for s in atoms.get_chemical_symbols()]
        >>> c = FixAtoms(mask=mask)
        >>> atoms.set_constraint(c)

        Fix all atoms with z-coordinate less than 1.0 Angstrom:

        >>> c = FixAtoms(mask=atoms.positions[:, 2] < 1.0)
        >>> atoms.set_constraint(c)
        """
        if indices is None and mask is None:
            raise ValueError('Use "indices" or "mask".')
        if indices is not None and mask is not None:
            raise ValueError('Use only one of "indices" and "mask".')
        if mask is not None:
            indices = np.arange(len(mask))[np.asarray(mask, bool)]
        else:
            srt = np.sort(indices)
            if (np.diff(srt) == 0).any():
                raise ValueError('FixAtoms: The indices array contained duplicates. Perhaps you wanted to specify a mask instead, but forgot the mask= keyword.')
        self.index = np.asarray(indices, int)
        if self.index.ndim != 1:
            raise ValueError('Wrong argument to FixAtoms class!')

    def get_removed_dof(self, atoms):
        return 3 * len(self.index)

    def adjust_positions(self, atoms, new):
        new[self.index] = atoms.positions[self.index]

    def adjust_forces(self, atoms, forces):
        forces[self.index] = 0.0

    def index_shuffle(self, atoms, ind):
        index = []
        for new, old in slice2enlist(ind, len(atoms)):
            if old in self.index:
                index.append(new)
        if len(index) == 0:
            raise IndexError('All indices in FixAtoms not part of slice')
        self.index = np.asarray(index, int)

    def get_indices(self):
        return self.index

    def __repr__(self):
        return 'FixAtoms(indices=%s)' % ints2string(self.index)

    def todict(self):
        return {'name': 'FixAtoms', 'kwargs': {'indices': self.index.tolist()}}

    def repeat(self, m, n):
        i0 = 0
        natoms = 0
        if isinstance(m, int):
            m = (m, m, m)
        index_new = []
        for m2 in range(m[2]):
            for m1 in range(m[1]):
                for m0 in range(m[0]):
                    i1 = i0 + n
                    index_new += [i + natoms for i in self.index]
                    i0 = i1
                    natoms += n
        self.index = np.asarray(index_new, int)
        return self

    def delete_atoms(self, indices, natoms):
        """Removes atom number ind from the index array, if present.

        Required for removing atoms with existing FixAtoms constraints.
        """
        i = np.zeros(natoms, int) - 1
        new = np.delete(np.arange(natoms), indices)
        i[new] = np.arange(len(new))
        index = i[self.index]
        self.index = index[index >= 0]
        if len(self.index) == 0:
            return None
        return self