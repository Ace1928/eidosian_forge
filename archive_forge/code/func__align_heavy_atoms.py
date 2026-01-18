from __future__ import annotations
import abc
import copy
import itertools
import logging
import math
import re
from typing import TYPE_CHECKING
import numpy as np
from monty.dev import requires
from monty.json import MSONable
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from pymatgen.core.structure import Molecule
@staticmethod
def _align_heavy_atoms(mol1, mol2, vmol1, vmol2, ilabel1, ilabel2, eq_atoms):
    """
        Align the label of topologically identical atoms of second molecule
        towards first molecule.

        Args:
            mol1: First molecule. OpenBabel OBMol object
            mol2: Second molecule. OpenBabel OBMol object
            vmol1: First virtual molecule constructed by centroids. OpenBabel
                OBMol object
            vmol2: First virtual molecule constructed by centroids. OpenBabel
                OBMol object
            ilabel1: inchi label map of the first molecule
            ilabel2: inchi label map of the second molecule
            eq_atoms: equivalent atom labels

        Returns:
            corrected inchi labels of heavy atoms of the second molecule
        """
    n_virtual = vmol1.NumAtoms()
    n_heavy = len(ilabel1)
    for idx in ilabel2:
        a1 = vmol1.NewAtom()
        a1.SetAtomicNum(1)
        a1.SetVector(0.0, 0.0, 0.0)
        oa2 = mol2.GetAtom(idx)
        a2 = vmol2.NewAtom()
        a2.SetAtomicNum(1)
        a2.SetVector(oa2.GetVector())
    aligner = openbabel.OBAlign(False, False)
    aligner.SetRefMol(vmol1)
    aligner.SetTargetMol(vmol2)
    aligner.Align()
    aligner.UpdateCoords(vmol2)
    canon_mol1 = openbabel.OBMol()
    for idx in ilabel1:
        oa1 = mol1.GetAtom(idx)
        a1 = canon_mol1.NewAtom()
        a1.SetAtomicNum(oa1.GetAtomicNum())
        a1.SetVector(oa1.GetVector())
    aligned_mol2 = openbabel.OBMol()
    for idx in range(n_virtual + 1, n_virtual + n_heavy + 1):
        oa2 = vmol2.GetAtom(idx)
        a2 = aligned_mol2.NewAtom()
        a2.SetAtomicNum(oa2.GetAtomicNum())
        a2.SetVector(oa2.GetVector())
    canon_label2 = list(range(1, n_heavy + 1))
    for symm in eq_atoms:
        for idx in symm:
            canon_label2[idx - 1] = -1
    for symm in eq_atoms:
        candidates1 = list(symm)
        candidates2 = list(symm)
        for c2 in candidates2:
            distance = 99999.0
            canon_idx = candidates1[0]
            a2 = aligned_mol2.GetAtom(c2)
            for c1 in candidates1:
                a1 = canon_mol1.GetAtom(c1)
                dist = a1.GetDistance(a2)
                if dist < distance:
                    distance = dist
                    canon_idx = c1
            canon_label2[c2 - 1] = canon_idx
            candidates1.remove(canon_idx)
    canon_inchi_orig_map2 = list(zip(canon_label2, list(range(1, n_heavy + 1)), ilabel2))
    canon_inchi_orig_map2.sort(key=lambda m: m[0])
    return tuple((x[2] for x in canon_inchi_orig_map2))