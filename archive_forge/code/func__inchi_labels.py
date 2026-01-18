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
def _inchi_labels(mol):
    """
        Get the inchi canonical labels of the heavy atoms in the molecule.

        Args:
            mol: The molecule. OpenBabel OBMol object

        Returns:
            The label mappings. List of tuple of canonical label,
            original label
            List of equivalent atoms.
        """
    ob_conv = openbabel.OBConversion()
    ob_conv.SetOutFormat('inchi')
    ob_conv.AddOption('a', openbabel.OBConversion.OUTOPTIONS)
    ob_conv.AddOption('X', openbabel.OBConversion.OUTOPTIONS, 'DoNotAddH')
    inchi_text = ob_conv.WriteString(mol)
    match = re.search('InChI=(?P<inchi>.+)\\nAuxInfo=.+/N:(?P<labels>[0-9,;]+)/(E:(?P<eq_atoms>[0-9,;\\(\\)]*)/)?', inchi_text)
    inchi = match.group('inchi')
    label_text = match.group('labels')
    eq_atom_text = match.group('eq_atoms')
    heavy_atom_labels = tuple((int(idx) for idx in label_text.replace(';', ',').split(',')))
    eq_atoms = []
    if eq_atom_text is not None:
        eq_tokens = re.findall('\\(((?:[0-9]+,)+[0-9]+)\\)', eq_atom_text.replace(';', ','))
        eq_atoms = tuple((tuple((int(idx) for idx in t.split(','))) for t in eq_tokens))
    return (heavy_atom_labels, eq_atoms, inchi)