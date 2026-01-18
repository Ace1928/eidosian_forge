from warnings import warn
import copy
import logging
from rdkit import Chem
from .utils import memoized_property
Neutralize molecule by adding/removing hydrogens. Attempts to preserve zwitterions.

        :param mol: The molecule to uncharge.
        :type mol: :rdkit:`Mol <Chem.rdchem.Mol-class.html>`
        :return: The uncharged molecule.
        :rtype: :rdkit:`Mol <Chem.rdchem.Mol-class.html>`
        