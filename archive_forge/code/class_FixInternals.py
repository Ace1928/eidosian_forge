from math import sqrt
from warnings import warn
import numpy as np
from scipy.linalg import expm, logm
from ase.calculators.calculator import PropertyNotImplementedError
from ase.geometry import (find_mic, wrap_positions, get_distances_derivatives,
from ase.utils.parsemath import eval_expression
from ase.stress import (full_3x3_to_voigt_6_stress,
class FixInternals(FixConstraint):
    """Constraint object for fixing multiple internal coordinates.

    Allows fixing bonds, angles, and dihedrals.
    Please provide angular units in degrees using angles_deg and
    dihedrals_deg.
    """

    def __init__(self, bonds=None, angles=None, dihedrals=None, angles_deg=None, dihedrals_deg=None, bondcombos=None, anglecombos=None, dihedralcombos=None, mic=False, epsilon=1e-07):
        warn_msg = 'Please specify {} in degrees using the {} argument.'
        if angles:
            warn(FutureWarning(warn_msg.format('angles', 'angle_deg')))
            angles = np.asarray(angles)
            angles[:, 0] = angles[:, 0] / np.pi * 180
            angles = angles.tolist()
        else:
            angles = angles_deg
        if dihedrals:
            warn(FutureWarning(warn_msg.format('dihedrals', 'dihedrals_deg')))
            dihedrals = np.asarray(dihedrals)
            dihedrals[:, 0] = dihedrals[:, 0] / np.pi * 180
            dihedrals = dihedrals.tolist()
        else:
            dihedrals = dihedrals_deg
        self.bonds = bonds or []
        self.angles = angles or []
        self.dihedrals = dihedrals or []
        self.bondcombos = bondcombos or []
        self.anglecombos = anglecombos or []
        self.dihedralcombos = dihedralcombos or []
        self.mic = mic
        self.epsilon = epsilon
        self.n = len(self.bonds) + len(self.angles) + len(self.dihedrals) + len(self.bondcombos) + len(self.anglecombos) + len(self.dihedralcombos)
        self.constraints = []
        self.initialized = False

    def get_removed_dof(self, atoms):
        return self.n

    def initialize(self, atoms):
        if self.initialized:
            return
        masses = np.repeat(atoms.get_masses(), 3)
        cell = None
        pbc = None
        if self.mic:
            cell = atoms.cell
            pbc = atoms.pbc
        self.constraints = []
        for data, make_constr in [(self.bonds, self.FixBondLengthAlt), (self.angles, self.FixAngle), (self.dihedrals, self.FixDihedral), (self.bondcombos, self.FixBondCombo), (self.anglecombos, self.FixAngleCombo), (self.dihedralcombos, self.FixDihedralCombo)]:
            for datum in data:
                constr = make_constr(datum[0], datum[1], masses, cell, pbc)
                self.constraints.append(constr)
        self.initialized = True

    def shuffle_definitions(self, shuffle_dic, internal_type):
        dfns = []
        for dfn in internal_type:
            append = True
            new_dfn = [dfn[0], list(dfn[1])]
            for old in dfn[1]:
                if old in shuffle_dic:
                    new_dfn[1][dfn[1].index(old)] = shuffle_dic[old]
                else:
                    append = False
                    break
            if append:
                dfns.append(new_dfn)
        return dfns

    def shuffle_combos(self, shuffle_dic, internal_type):
        dfns = []
        for dfn in internal_type:
            append = True
            all_indices = [idx[0:-1] for idx in dfn[1]]
            new_dfn = [dfn[0], list(dfn[1])]
            for i, indices in enumerate(all_indices):
                for old in indices:
                    if old in shuffle_dic:
                        new_dfn[1][i][indices.index(old)] = shuffle_dic[old]
                    else:
                        append = False
                        break
                if not append:
                    break
            if append:
                dfns.append(new_dfn)
        return dfns

    def index_shuffle(self, atoms, ind):
        self.initialize(atoms)
        shuffle_dic = dict(slice2enlist(ind, len(atoms)))
        shuffle_dic = {old: new for new, old in shuffle_dic.items()}
        self.bonds = self.shuffle_definitions(shuffle_dic, self.bonds)
        self.angles = self.shuffle_definitions(shuffle_dic, self.angles)
        self.dihedrals = self.shuffle_definitions(shuffle_dic, self.dihedrals)
        self.bondcombos = self.shuffle_combos(shuffle_dic, self.bondcombos)
        self.anglecombos = self.shuffle_combos(shuffle_dic, self.anglecombos)
        self.dihedralcombos = self.shuffle_combos(shuffle_dic, self.dihedralcombos)
        self.initialized = False
        self.initialize(atoms)
        if len(self.constraints) == 0:
            raise IndexError('Constraint not part of slice')

    def get_indices(self):
        cons = []
        for dfn in self.bonds + self.dihedrals + self.angles:
            cons.extend(dfn[1])
        for dfn in self.bondcombos + self.anglecombos + self.dihedralcombos:
            for partial_dfn in dfn[1]:
                cons.extend(partial_dfn[0:-1])
        return list(set(cons))

    def todict(self):
        return {'name': 'FixInternals', 'kwargs': {'bonds': self.bonds, 'angles': self.angles, 'dihedrals': self.dihedrals, 'bondcombos': self.bondcombos, 'anglecombos': self.anglecombos, 'dihedralcombos': self.dihedralcombos, 'mic': self.mic, 'epsilon': self.epsilon}}

    def adjust_positions(self, atoms, new):
        self.initialize(atoms)
        for constraint in self.constraints:
            constraint.prepare_jacobian(atoms.positions)
        for j in range(50):
            maxerr = 0.0
            for constraint in self.constraints:
                constraint.adjust_positions(atoms.positions, new)
                maxerr = max(abs(constraint.sigma), maxerr)
            if maxerr < self.epsilon:
                return
        raise ValueError('Shake did not converge.')

    def adjust_forces(self, atoms, forces):
        """Project out translations and rotations and all other constraints"""
        self.initialize(atoms)
        positions = atoms.positions
        N = len(forces)
        list2_constraints = list(np.zeros((6, N, 3)))
        tx, ty, tz, rx, ry, rz = list2_constraints
        list_constraints = [r.ravel() for r in list2_constraints]
        tx[:, 0] = 1.0
        ty[:, 1] = 1.0
        tz[:, 2] = 1.0
        ff = forces.ravel()
        center = positions.sum(axis=0) / N
        rx[:, 1] = -(positions[:, 2] - center[2])
        rx[:, 2] = positions[:, 1] - center[1]
        ry[:, 0] = positions[:, 2] - center[2]
        ry[:, 2] = -(positions[:, 0] - center[0])
        rz[:, 0] = -(positions[:, 1] - center[1])
        rz[:, 1] = positions[:, 0] - center[0]
        for r in list2_constraints:
            r /= np.linalg.norm(r.ravel())
        for constraint in self.constraints:
            constraint.prepare_jacobian(positions)
            constraint.adjust_forces(positions, forces)
            list_constraints.insert(0, constraint.jacobian)
        list_constraints = [r.ravel() for r in list_constraints]
        aa = np.column_stack(list_constraints)
        aa, bb = np.linalg.qr(aa)
        hh = []
        for i, constraint in enumerate(self.constraints):
            hh.append(aa[:, i] * np.row_stack(aa[:, i]))
        txx = aa[:, self.n] * np.row_stack(aa[:, self.n])
        tyy = aa[:, self.n + 1] * np.row_stack(aa[:, self.n + 1])
        tzz = aa[:, self.n + 2] * np.row_stack(aa[:, self.n + 2])
        rxx = aa[:, self.n + 3] * np.row_stack(aa[:, self.n + 3])
        ryy = aa[:, self.n + 4] * np.row_stack(aa[:, self.n + 4])
        rzz = aa[:, self.n + 5] * np.row_stack(aa[:, self.n + 5])
        T = txx + tyy + tzz + rxx + ryy + rzz
        for vec in hh:
            T += vec
        ff = np.dot(T, np.row_stack(ff))
        forces[:, :] -= np.dot(T, np.row_stack(ff)).reshape(-1, 3)

    def __repr__(self):
        constraints = repr(self.constraints)
        return 'FixInternals(_copy_init=%s, epsilon=%s)' % (constraints, repr(self.epsilon))

    def __str__(self):
        return '\n'.join([repr(c) for c in self.constraints])

    class FixInternalsBase:
        """Base class for subclasses of FixInternals."""

        def __init__(self, targetvalue, indices, masses, cell, pbc):
            self.targetvalue = targetvalue
            self.indices = [defin[0:-1] for defin in indices]
            self.coefs = np.asarray([defin[-1] for defin in indices])
            self.masses = masses
            self.jacobian = []
            self.sigma = 1.0
            self.projected_force = None
            self.cell = cell
            self.pbc = pbc

        def finalize_jacobian(self, pos, n_internals, n, derivs):
            """Populate jacobian with derivatives for `n_internals` defined
            internals. n = 2 (bonds), 3 (angles), 4 (dihedrals)."""
            jacobian = np.zeros((n_internals, *pos.shape))
            for i, idx in enumerate(self.indices):
                for j in range(n):
                    jacobian[i, idx[j]] = derivs[i, j]
            jacobian = jacobian.reshape((n_internals, 3 * len(pos)))
            self.jacobian = self.coefs @ jacobian

        def finalize_positions(self, newpos):
            jacobian = self.jacobian / self.masses
            lamda = -self.sigma / np.dot(jacobian, self.jacobian)
            dnewpos = lamda * jacobian
            newpos += dnewpos.reshape(newpos.shape)

        def adjust_forces(self, positions, forces):
            self.projected_force = np.dot(self.jacobian, forces.ravel())
            self.jacobian /= np.linalg.norm(self.jacobian)

    class FixBondCombo(FixInternalsBase):
        """Constraint subobject for fixing linear combination of bond lengths
        within FixInternals.

        sum_i( coef_i * bond_length_i ) = constant
        """

        def prepare_jacobian(self, pos):
            bondvectors = [pos[k] - pos[h] for h, k in self.indices]
            derivs = get_distances_derivatives(bondvectors, cell=self.cell, pbc=self.pbc)
            self.finalize_jacobian(pos, len(bondvectors), 2, derivs)

        def adjust_positions(self, oldpos, newpos):
            bondvectors = [newpos[k] - newpos[h] for h, k in self.indices]
            (_,), (dists,) = conditional_find_mic([bondvectors], cell=self.cell, pbc=self.pbc)
            value = np.dot(self.coefs, dists)
            self.sigma = value - self.targetvalue
            self.finalize_positions(newpos)

        def __repr__(self):
            return 'FixBondCombo({}, {}, {})'.format(repr(self.targetvalue), self.indices, self.coefs)

    class FixBondLengthAlt(FixBondCombo):
        """Constraint subobject for fixing bond length within FixInternals.
        Fix distance between atoms with indices a1, a2."""

        def __init__(self, targetvalue, indices, masses, cell, pbc):
            indices = [list(indices) + [1.0]]
            super().__init__(targetvalue, indices, masses, cell=cell, pbc=pbc)

        def __repr__(self):
            return 'FixBondLengthAlt({}, {})'.format(self.targetvalue, *self.indices)

    class FixAngleCombo(FixInternalsBase):
        """Constraint subobject for fixing linear combination of angles
        within FixInternals.

        sum_i( coef_i * angle_i ) = constant
        """

        def gather_vectors(self, pos):
            v0 = [pos[h] - pos[k] for h, k, l in self.indices]
            v1 = [pos[l] - pos[k] for h, k, l in self.indices]
            return (v0, v1)

        def prepare_jacobian(self, pos):
            v0, v1 = self.gather_vectors(pos)
            derivs = get_angles_derivatives(v0, v1, cell=self.cell, pbc=self.pbc)
            self.finalize_jacobian(pos, len(v0), 3, derivs)

        def adjust_positions(self, oldpos, newpos):
            v0, v1 = self.gather_vectors(newpos)
            value = get_angles(v0, v1, cell=self.cell, pbc=self.pbc)
            value = np.dot(self.coefs, value)
            self.sigma = value - self.targetvalue
            self.finalize_positions(newpos)

        def __repr__(self):
            return 'FixAngleCombo({}, {}, {})'.format(self.targetvalue, self.indices, self.coefs)

    class FixAngle(FixAngleCombo):
        """Constraint object for fixing an angle within
        FixInternals using the SHAKE algorithm.

        SHAKE convergence is potentially problematic for angles very close to
        0 or 180 degrees as there is a singularity in the Cartesian derivative.
        """

        def __init__(self, targetvalue, indices, masses, cell, pbc):
            """Fix atom movement to construct a constant angle."""
            indices = [list(indices) + [1.0]]
            super().__init__(targetvalue, indices, masses, cell=cell, pbc=pbc)

        def __repr__(self):
            return 'FixAngle({}, {})'.format(self.targetvalue, *self.indices)

    class FixDihedralCombo(FixInternalsBase):
        """Constraint subobject for fixing linear combination of dihedrals
        within FixInternals.

        sum_i( coef_i * dihedral_i ) = constant
        """

        def gather_vectors(self, pos):
            v0 = [pos[k] - pos[h] for h, k, l, m in self.indices]
            v1 = [pos[l] - pos[k] for h, k, l, m in self.indices]
            v2 = [pos[m] - pos[l] for h, k, l, m in self.indices]
            return (v0, v1, v2)

        def prepare_jacobian(self, pos):
            v0, v1, v2 = self.gather_vectors(pos)
            derivs = get_dihedrals_derivatives(v0, v1, v2, cell=self.cell, pbc=self.pbc)
            self.finalize_jacobian(pos, len(v0), 4, derivs)

        def adjust_positions(self, oldpos, newpos):
            v0, v1, v2 = self.gather_vectors(newpos)
            value = get_dihedrals(v0, v1, v2, cell=self.cell, pbc=self.pbc)
            value = np.dot(self.coefs, value)
            self.sigma = value - self.targetvalue
            self.finalize_positions(newpos)

        def __repr__(self):
            return 'FixDihedralCombo({}, {}, {})'.format(self.targetvalue, self.indices, self.coefs)

    class FixDihedral(FixDihedralCombo):
        """Constraint object for fixing a dihedral angle using
        the SHAKE algorithm. This one allows also other constraints.

        SHAKE convergence is potentially problematic for near-undefined
        dihedral angles (i.e. when one of the two angles a012 or a123
        approaches 0 or 180 degrees).
        """

        def __init__(self, targetvalue, indices, masses, cell, pbc):
            indices = [list(indices) + [1.0]]
            super().__init__(targetvalue, indices, masses, cell=cell, pbc=pbc)

        def adjust_positions(self, oldpos, newpos):
            v0, v1, v2 = self.gather_vectors(newpos)
            value = get_dihedrals(v0, v1, v2, cell=self.cell, pbc=self.pbc)
            self.sigma = (value - self.targetvalue + 180) % 360 - 180
            self.finalize_positions(newpos)

        def __repr__(self):
            return 'FixDihedral({}, {})'.format(self.targetvalue, *self.indices)