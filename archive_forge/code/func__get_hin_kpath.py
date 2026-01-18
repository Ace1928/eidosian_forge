from __future__ import annotations
import itertools
from warnings import warn
import networkx as nx
import numpy as np
from pymatgen.electronic_structure.bandstructure import BandStructureSymmLine
from pymatgen.electronic_structure.core import Spin
from pymatgen.symmetry.analyzer import cite_conventional_cell_algo
from pymatgen.symmetry.kpath import KPathBase, KPathLatimerMunro, KPathSeek, KPathSetyawanCurtarolo
def _get_hin_kpath(self, symprec, angle_tolerance, atol, tri):
    """
        Returns:
            Hinuma et al. k-path with labels.
        """
    bs = KPathSeek(self._structure, symprec, angle_tolerance, atol, tri)
    kpoints = bs.kpath['kpoints']
    tmat = bs._tmat
    for key in kpoints:
        kpoints[key] = np.dot(np.transpose(np.linalg.inv(tmat)), kpoints[key])
    bs.kpath['kpoints'] = kpoints
    self._rec_lattice = self._structure.lattice.reciprocal_lattice
    warn('K-path from the Hinuma et al. convention has been transformed to the basis of the reciprocal latticeof the input structure. Use `KPathSeek` for the path in the original author-intended basis.')
    return bs