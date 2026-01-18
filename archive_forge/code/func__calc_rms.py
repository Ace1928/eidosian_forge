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
def _calc_rms(mol1, mol2, clabel1, clabel2):
    """
        Calculate the RMSD.

        Args:
            mol1: The first molecule. OpenBabel OBMol or pymatgen Molecule
                object
            mol2: The second molecule. OpenBabel OBMol or pymatgen Molecule
                object
            clabel1: The atom indices that can reorder the first molecule to
                uniform atom order
            clabel1: The atom indices that can reorder the second molecule to
                uniform atom order

        Returns:
            The RMSD.
        """
    ob_mol1 = BabelMolAdaptor(mol1).openbabel_mol
    ob_mol2 = BabelMolAdaptor(mol2).openbabel_mol
    cmol1 = openbabel.OBMol()
    for idx in clabel1:
        oa1 = ob_mol1.GetAtom(idx)
        a1 = cmol1.NewAtom()
        a1.SetAtomicNum(oa1.GetAtomicNum())
        a1.SetVector(oa1.GetVector())
    cmol2 = openbabel.OBMol()
    for idx in clabel2:
        oa2 = ob_mol2.GetAtom(idx)
        a2 = cmol2.NewAtom()
        a2.SetAtomicNum(oa2.GetAtomicNum())
        a2.SetVector(oa2.GetVector())
    aligner = openbabel.OBAlign(True, False)
    aligner.SetRefMol(cmol1)
    aligner.SetTargetMol(cmol2)
    aligner.Align()
    return aligner.GetRMSD()