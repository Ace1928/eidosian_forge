from math import sqrt
from warnings import warn
import numpy as np
from scipy.linalg import expm, logm
from ase.calculators.calculator import PropertyNotImplementedError
from ase.geometry import (find_mic, wrap_positions, get_distances_derivatives,
from ase.utils.parsemath import eval_expression
from ase.stress import (full_3x3_to_voigt_6_stress,
class UnitCellFilter(Filter):
    """Modify the supercell and the atom positions. """

    def __init__(self, atoms, mask=None, cell_factor=None, hydrostatic_strain=False, constant_volume=False, scalar_pressure=0.0):
        """Create a filter that returns the atomic forces and unit cell
        stresses together, so they can simultaneously be minimized.

        The first argument, atoms, is the atoms object. The optional second
        argument, mask, is a list of booleans, indicating which of the six
        independent components of the strain are relaxed.

        - True = relax to zero
        - False = fixed, ignore this component

        Degrees of freedom are the positions in the original undeformed cell,
        plus the deformation tensor (extra 3 "atoms"). This gives forces
        consistent with numerical derivatives of the potential energy
        with respect to the cell degreees of freedom.

        For full details see:
            E. B. Tadmor, G. S. Smith, N. Bernstein, and E. Kaxiras,
            Phys. Rev. B 59, 235 (1999)

        You can still use constraints on the atoms, e.g. FixAtoms, to control
        the relaxation of the atoms.

        >>> # this should be equivalent to the StrainFilter
        >>> atoms = Atoms(...)
        >>> atoms.set_constraint(FixAtoms(mask=[True for atom in atoms]))
        >>> ucf = UnitCellFilter(atoms)

        You should not attach this UnitCellFilter object to a
        trajectory. Instead, create a trajectory for the atoms, and
        attach it to an optimizer like this:

        >>> atoms = Atoms(...)
        >>> ucf = UnitCellFilter(atoms)
        >>> qn = QuasiNewton(ucf)
        >>> traj = Trajectory('TiO2.traj', 'w', atoms)
        >>> qn.attach(traj)
        >>> qn.run(fmax=0.05)

        Helpful conversion table:

        - 0.05 eV/A^3   = 8 GPA
        - 0.003 eV/A^3  = 0.48 GPa
        - 0.0006 eV/A^3 = 0.096 GPa
        - 0.0003 eV/A^3 = 0.048 GPa
        - 0.0001 eV/A^3 = 0.02 GPa

        Additional optional arguments:

        cell_factor: float (default float(len(atoms)))
            Factor by which deformation gradient is multiplied to put
            it on the same scale as the positions when assembling
            the combined position/cell vector. The stress contribution to
            the forces is scaled down by the same factor. This can be thought
            of as a very simple preconditioners. Default is number of atoms
            which gives approximately the correct scaling.

        hydrostatic_strain: bool (default False)
            Constrain the cell by only allowing hydrostatic deformation.
            The virial tensor is replaced by np.diag([np.trace(virial)]*3).

        constant_volume: bool (default False)
            Project out the diagonal elements of the virial tensor to allow
            relaxations at constant volume, e.g. for mapping out an
            energy-volume curve. Note: this only approximately conserves
            the volume and breaks energy/force consistency so can only be
            used with optimizers that do require do a line minimisation
            (e.g. FIRE).

        scalar_pressure: float (default 0.0)
            Applied pressure to use for enthalpy pV term. As above, this
            breaks energy/force consistency.
        """
        Filter.__init__(self, atoms, indices=range(len(atoms)))
        self.atoms = atoms
        self.orig_cell = atoms.get_cell()
        self.stress = None
        if mask is None:
            mask = np.ones(6)
        mask = np.asarray(mask)
        if mask.shape == (6,):
            self.mask = voigt_6_to_full_3x3_stress(mask)
        elif mask.shape == (3, 3):
            self.mask = mask
        else:
            raise ValueError('shape of mask should be (3,3) or (6,)')
        if cell_factor is None:
            cell_factor = float(len(atoms))
        self.hydrostatic_strain = hydrostatic_strain
        self.constant_volume = constant_volume
        self.scalar_pressure = scalar_pressure
        self.cell_factor = cell_factor
        self.copy = self.atoms.copy
        self.arrays = self.atoms.arrays

    def deform_grad(self):
        return np.linalg.solve(self.orig_cell, self.atoms.cell).T

    def get_positions(self):
        """
        this returns an array with shape (natoms + 3,3).

        the first natoms rows are the positions of the atoms, the last
        three rows are the deformation tensor associated with the unit cell,
        scaled by self.cell_factor.
        """
        cur_deform_grad = self.deform_grad()
        natoms = len(self.atoms)
        pos = np.zeros((natoms + 3, 3))
        pos[:natoms] = np.linalg.solve(cur_deform_grad, self.atoms.positions.T).T
        pos[natoms:] = self.cell_factor * cur_deform_grad
        return pos

    def set_positions(self, new, **kwargs):
        """
        new is an array with shape (natoms+3,3).

        the first natoms rows are the positions of the atoms, the last
        three rows are the deformation tensor used to change the cell shape.

        the new cell is first set from original cell transformed by the new
        deformation gradient, then the positions are set with respect to the
        current cell by transforming them with the same deformation gradient
        """
        natoms = len(self.atoms)
        new_atom_positions = new[:natoms]
        new_deform_grad = new[natoms:] / self.cell_factor
        self.atoms.set_cell(self.orig_cell @ new_deform_grad.T, scale_atoms=True)
        self.atoms.set_positions(new_atom_positions @ new_deform_grad.T, **kwargs)

    def get_potential_energy(self, force_consistent=True):
        """
        returns potential energy including enthalpy PV term.
        """
        atoms_energy = self.atoms.get_potential_energy(force_consistent=force_consistent)
        return atoms_energy + self.scalar_pressure * self.atoms.get_volume()

    def get_forces(self, **kwargs):
        """
        returns an array with shape (natoms+3,3) of the atomic forces
        and unit cell stresses.

        the first natoms rows are the forces on the atoms, the last
        three rows are the forces on the unit cell, which are
        computed from the stress tensor.
        """
        stress = self.atoms.get_stress(**kwargs)
        atoms_forces = self.atoms.get_forces(**kwargs)
        volume = self.atoms.get_volume()
        virial = -volume * (voigt_6_to_full_3x3_stress(stress) + np.diag([self.scalar_pressure] * 3))
        cur_deform_grad = self.deform_grad()
        atoms_forces = atoms_forces @ cur_deform_grad
        virial = np.linalg.solve(cur_deform_grad, virial.T).T
        if self.hydrostatic_strain:
            vtr = virial.trace()
            virial = np.diag([vtr / 3.0, vtr / 3.0, vtr / 3.0])
        if (self.mask != 1.0).any():
            virial *= self.mask
        if self.constant_volume:
            vtr = virial.trace()
            np.fill_diagonal(virial, np.diag(virial) - vtr / 3.0)
        natoms = len(self.atoms)
        forces = np.zeros((natoms + 3, 3))
        forces[:natoms] = atoms_forces
        forces[natoms:] = virial / self.cell_factor
        self.stress = -full_3x3_to_voigt_6_stress(virial) / volume
        return forces

    def get_stress(self):
        raise PropertyNotImplementedError

    def has(self, x):
        return self.atoms.has(x)

    def __len__(self):
        return len(self.atoms) + 3