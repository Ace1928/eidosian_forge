from __future__ import annotations
import copy
import logging
import math
import os
import re
import struct
import warnings
from typing import TYPE_CHECKING, Any
import networkx as nx
import numpy as np
import pandas as pd
from monty.io import zopen
from monty.json import MSONable, jsanitize
from pymatgen.analysis.graphs import MoleculeGraph
from pymatgen.analysis.local_env import OpenBabelNN
from pymatgen.core import Molecule
from pymatgen.io.qchem.utils import (
def _read_eigenvalues(self):
    """Parse the orbital energies from the output file. An array of the
        dimensions of the number of orbitals used in the calculation is stored.
        """
    header_pattern = 'Final Alpha MO Eigenvalues'
    elements_pattern = '\\-*\\d+\\.\\d+'
    if not self.data.get('unrestricted', []):
        spin_unrestricted = False
        footer_pattern = 'Final Alpha MO Coefficients+\\s*'
    else:
        spin_unrestricted = True
        footer_pattern = 'Final Beta MO Eigenvalues'
    alpha_eigenvalues = read_matrix_pattern(header_pattern, footer_pattern, elements_pattern, self.text, postprocess=float)
    if spin_unrestricted:
        header_pattern = 'Final Beta MO Eigenvalues'
        footer_pattern = 'Final Alpha MO Coefficients+\\s*'
        beta_eigenvalues = read_matrix_pattern(header_pattern, footer_pattern, elements_pattern, self.text, postprocess=float)
    self.data['alpha_eigenvalues'] = alpha_eigenvalues
    if spin_unrestricted:
        self.data['beta_eigenvalues'] = beta_eigenvalues