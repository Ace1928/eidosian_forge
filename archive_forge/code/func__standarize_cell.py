from collections import Counter
from itertools import combinations, product, filterfalse
import numpy as np
from scipy.spatial import cKDTree as KDTree
from ase import Atom, Atoms
from ase.build.tools import niggli_reduce
def _standarize_cell(self, atoms):
    """Rotate the first vector such that it points along the x-axis.
        Then rotate around the first vector so the second vector is in the
        xy plane.
        """
    cell = atoms.get_cell().T
    total_rot_mat = np.eye(3)
    v1 = cell[:, 0]
    l1 = np.sqrt(v1[0] ** 2 + v1[2] ** 2)
    angle = np.abs(np.arcsin(v1[2] / l1))
    if v1[0] < 0.0 and v1[2] > 0.0:
        angle = np.pi - angle
    elif v1[0] < 0.0 and v1[2] < 0.0:
        angle = np.pi + angle
    elif v1[0] > 0.0 and v1[2] < 0.0:
        angle = -angle
    ca = np.cos(angle)
    sa = np.sin(angle)
    rotmat = np.array([[ca, 0.0, sa], [0.0, 1.0, 0.0], [-sa, 0.0, ca]])
    total_rot_mat = rotmat.dot(total_rot_mat)
    cell = rotmat.dot(cell)
    v1 = cell[:, 0]
    l1 = np.sqrt(v1[0] ** 2 + v1[1] ** 2)
    angle = np.abs(np.arcsin(v1[1] / l1))
    if v1[0] < 0.0 and v1[1] > 0.0:
        angle = np.pi - angle
    elif v1[0] < 0.0 and v1[1] < 0.0:
        angle = np.pi + angle
    elif v1[0] > 0.0 and v1[1] < 0.0:
        angle = -angle
    ca = np.cos(angle)
    sa = np.sin(angle)
    rotmat = np.array([[ca, sa, 0.0], [-sa, ca, 0.0], [0.0, 0.0, 1.0]])
    total_rot_mat = rotmat.dot(total_rot_mat)
    cell = rotmat.dot(cell)
    v2 = cell[:, 1]
    l2 = np.sqrt(v2[1] ** 2 + v2[2] ** 2)
    angle = np.abs(np.arcsin(v2[2] / l2))
    if v2[1] < 0.0 and v2[2] > 0.0:
        angle = np.pi - angle
    elif v2[1] < 0.0 and v2[2] < 0.0:
        angle = np.pi + angle
    elif v2[1] > 0.0 and v2[2] < 0.0:
        angle = -angle
    ca = np.cos(angle)
    sa = np.sin(angle)
    rotmat = np.array([[1.0, 0.0, 0.0], [0.0, ca, sa], [0.0, -sa, ca]])
    total_rot_mat = rotmat.dot(total_rot_mat)
    cell = rotmat.dot(cell)
    atoms.set_cell(cell.T)
    atoms.set_positions(total_rot_mat.dot(atoms.get_positions().T).T)
    atoms.wrap(pbc=[1, 1, 1])
    return atoms