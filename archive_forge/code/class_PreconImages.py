import sys
import time
import copy
import warnings
from abc import ABC, abstractmethod
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.interpolate import CubicSpline
from ase.constraints import Filter, FixAtoms
from ase.utils import longsum
from ase.geometry import find_mic
import ase.utils.ff as ff
import ase.units as units
from ase.optimize.precon.neighbors import (get_neighbours,
from ase.neighborlist import neighbor_list
class PreconImages:

    def __init__(self, precon, images, **kwargs):
        """
        Wrapper for a list of Precon objects and associated images
    
        This is used when preconditioning a NEB object. Equation references
        refer to Paper IV in the :class:`ase.neb.NEB` documentation, i.e.
    
        S. Makri, C. Ortner and J. R. Kermode, J. Chem. Phys.
        150, 094109 (2019)
        https://dx.doi.org/10.1063/1.5064465

        Args:
            precon (str or list): preconditioner(s) to use
            images (list of Atoms): Atoms objects that define the state

        """
        self.images = images
        if isinstance(precon, list):
            if len(precon) != len(images):
                raise ValueError(f'length mismatch: len(precon)={len(precon)} != len(images)={len(images)}')
            self.precon = precon
            return
        P0 = make_precon(precon, images[0], **kwargs)
        self.precon = [P0]
        for image in images[1:]:
            P = P0.copy()
            P.make_precon(image)
            self.precon.append(P)
        self._spline = None

    def __len__(self):
        return len(self.precon)

    def __iter__(self):
        return iter(self.precon)

    def __getitem__(self, index):
        return self.precon[index]

    def apply(self, all_forces, index=None):
        """Apply preconditioners to stored images

        Args:
            all_forces (array): forces on images, shape (nimages, natoms, 3)
            index (slice, optional): Which images to include. Defaults to all.

        Returns:
            precon_forces: array of preconditioned forces
        """
        if index is None:
            index = slice(None)
        precon_forces = []
        for precon, image, forces in zip(self.precon[index], self.images[index], all_forces):
            f_vec = forces.reshape(-1)
            pf_vec, _ = precon.apply(f_vec, image)
            precon_forces.append(pf_vec.reshape(-1, 3))
        return np.array(precon_forces)

    def average_norm(self, i, j, dx):
        """Average norm between images i and j

        Args:
            i (int): left image
            j (int): right image
            dx (array): vector

        Returns:
            norm: norm of vector wrt average of precons at i and j
        """
        return np.sqrt(0.5 * (self.precon[i].dot(dx, dx) + self.precon[j].dot(dx, dx)))

    def get_tangent(self, i):
        """Normalised tangent vector at image i

        Args:
            i (int): image of interest

        Returns:
            tangent: tangent vector, normalised with appropriate precon norm
        """
        tangent = self.spline.dx_ds(self.spline.s[i])
        tangent /= self.precon[i].norm(tangent)
        return tangent.reshape(-1, 3)

    def get_residual(self, i, imgforce):
        P_dot_imgforce = self.precon[i].Pdot(imgforce.reshape(-1))
        return np.linalg.norm(P_dot_imgforce, np.inf)

    def get_spring_force(self, i, k1, k2, tangent):
        """Spring force on image

        Args:
            i (int): image of interest
            k1 (float): spring constant for left spring
            k2 (float): spring constant for right spring
            tangent (array): tangent vector, shape (natoms, 3)

        Returns:
            eta: NEB spring forces, shape (natoms, 3)
        """
        nimages = len(self.images)
        k = 0.5 * (k1 + k2) / nimages ** 2
        curvature = self.spline.d2x_ds2(self.spline.s[i]).reshape(-1, 3)
        eta = k * self.precon[i].vdot(curvature, tangent) * tangent
        return eta

    def get_coordinates(self, positions=None):
        """Compute displacements wrt appropriate precon metric for each image
        
        Args:
            positions (list or array, optional) - images positions.
                Shape either (nimages * natoms, 3) or ((nimages-2)*natoms, 3)

        Returns:
            s : array shape (nimages,), reaction coordinates, in range [0, 1]
            x : array shape (nimages, 3 * natoms), flat displacement vectors
        """
        nimages = len(self.precon)
        natoms = len(self.images[0])
        d_P = np.zeros(nimages)
        x = np.zeros((nimages, 3 * natoms))
        if positions is None:
            positions = [image.positions for image in self.images]
        elif isinstance(positions, np.ndarray) and len(positions.shape) == 2:
            positions = positions.reshape(-1, natoms, 3)
            positions = [positions[i, :, :] for i in range(len(positions))]
            if len(positions) == len(self.images) - 2:
                positions = [self.images[0].positions] + positions + [self.images[-1].positions]
        assert len(positions) == len(self.images)
        x[0, :] = positions[0].reshape(-1)
        for i in range(1, nimages):
            x[i, :] = positions[i].reshape(-1)
            dx, _ = find_mic(positions[i] - positions[i - 1], self.images[i - 1].cell, self.images[i - 1].pbc)
            dx = dx.reshape(-1)
            d_P[i] = self.average_norm(i, i - 1, dx)
        s = d_P.cumsum() / d_P.sum()
        return (s, x)

    def spline_fit(self, positions=None):
        """Fit 3 * natoms cubic splines as a function of reaction coordinate

        Returns:
            fit : :class:`ase.optimize.precon.SplineFit` object
        """
        s, x = self.get_coordinates(positions)
        return SplineFit(s, x)

    @property
    def spline(self):
        s, x = self.get_coordinates()
        if self._spline and (np.abs(s - self._old_s).max() < 1e-06 and np.abs(x - self._old_x).max() < 1e-06):
            return self._spline
        self._spline = self.spline_fit()
        self._old_s = s
        self._old_x = x
        return self._spline