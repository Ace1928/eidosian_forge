import numpy as np
from math import sqrt
from itertools import islice
from ase.io.formats import string2index
from ase.utils import rotate
from ase.data import covalent_radii, atomic_numbers
from ase.data.colors import jmol_colors
class PlottingVariables:

    def __init__(self, atoms, rotation='', show_unit_cell=2, radii=None, bbox=None, colors=None, scale=20, maxwidth=500, extra_offset=(0.0, 0.0)):
        self.numbers = atoms.get_atomic_numbers()
        self.colors = colors
        if colors is None:
            ncolors = len(jmol_colors)
            self.colors = jmol_colors[self.numbers.clip(max=ncolors - 1)]
        if radii is None:
            radii = covalent_radii[self.numbers]
        elif isinstance(radii, float):
            radii = covalent_radii[self.numbers] * radii
        else:
            radii = np.array(radii)
        natoms = len(atoms)
        if isinstance(rotation, str):
            rotation = rotate(rotation)
        cell = atoms.get_cell()
        disp = atoms.get_celldisp().flatten()
        if show_unit_cell > 0:
            L, T, D = cell_to_lines(self, cell)
            cell_vertices = np.empty((2, 2, 2, 3))
            for c1 in range(2):
                for c2 in range(2):
                    for c3 in range(2):
                        cell_vertices[c1, c2, c3] = np.dot([c1, c2, c3], cell) + disp
            cell_vertices.shape = (8, 3)
            cell_vertices = np.dot(cell_vertices, rotation)
        else:
            L = np.empty((0, 3))
            T = None
            D = None
            cell_vertices = None
        nlines = len(L)
        positions = np.empty((natoms + nlines, 3))
        R = atoms.get_positions()
        positions[:natoms] = R
        positions[natoms:] = L
        r2 = radii ** 2
        for n in range(nlines):
            d = D[T[n]]
            if ((((R - L[n] - d) ** 2).sum(1) < r2) & (((R - L[n] + d) ** 2).sum(1) < r2)).any():
                T[n] = -1
        positions = np.dot(positions, rotation)
        R = positions[:natoms]
        if bbox is None:
            X1 = (R - radii[:, None]).min(0)
            X2 = (R + radii[:, None]).max(0)
            if show_unit_cell == 2:
                X1 = np.minimum(X1, cell_vertices.min(0))
                X2 = np.maximum(X2, cell_vertices.max(0))
            M = (X1 + X2) / 2
            S = 1.05 * (X2 - X1)
            w = scale * S[0]
            if w > maxwidth:
                w = maxwidth
                scale = w / S[0]
            h = scale * S[1]
            offset = np.array([scale * M[0] - w / 2, scale * M[1] - h / 2, 0])
        else:
            w = (bbox[2] - bbox[0]) * scale
            h = (bbox[3] - bbox[1]) * scale
            offset = np.array([bbox[0], bbox[1], 0]) * scale
        offset[0] = offset[0] - extra_offset[0]
        offset[1] = offset[1] - extra_offset[1]
        self.w = w + extra_offset[0]
        self.h = h + extra_offset[1]
        positions *= scale
        positions -= offset
        if nlines > 0:
            D = np.dot(D, rotation)[:, :2] * scale
        if cell_vertices is not None:
            cell_vertices *= scale
            cell_vertices -= offset
        cell = np.dot(cell, rotation)
        cell *= scale
        self.cell = cell
        self.positions = positions
        self.D = D
        self.T = T
        self.cell_vertices = cell_vertices
        self.natoms = natoms
        self.d = 2 * scale * radii
        self.constraints = atoms.constraints
        self.frac_occ = False
        self.tags = None
        self.occs = None
        try:
            self.occs = atoms.info['occupancy']
            self.tags = atoms.get_tags()
            self.frac_occ = True
        except KeyError:
            pass