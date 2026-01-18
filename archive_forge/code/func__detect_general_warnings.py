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
def _detect_general_warnings(self):
    temp_inac_integ = read_pattern(self.text, {'key': 'Inaccurate integrated density:\\n\\s+Number of electrons\\s+=\\s+([\\d\\-\\.]+)\\n\\s+Numerical integral\\s+=\\s+([\\d\\-\\.]+)\\n\\s+Relative error\\s+=\\s+([\\d\\-\\.]+)\\s+\\%\\n'}).get('key')
    if temp_inac_integ is not None:
        inaccurate_integrated_density = np.zeros(shape=(len(temp_inac_integ), 3))
        for ii, entry in enumerate(temp_inac_integ):
            for jj, val in enumerate(entry):
                inaccurate_integrated_density[ii][jj] = float(val)
        self.data['warnings']['inaccurate_integrated_density'] = inaccurate_integrated_density
    if read_pattern(self.text, {'key': 'Intel MKL ERROR'}, terminate_on_match=True).get('key') == [[]]:
        self.data['warnings']['mkl'] = True
    if read_pattern(self.text, {'key': 'Starting finite difference calculation for IDERIV'}, terminate_on_match=True).get('key') == [[]]:
        self.data['warnings']['missing_analytical_derivates'] = True
    if read_pattern(self.text, {'key': 'Inconsistent size for SCF MO coefficient file'}, terminate_on_match=True).get('key') == [[]]:
        self.data['warnings']['inconsistent_size'] = True
    if read_pattern(self.text, {'key': 'Linear dependence detected in AO basis'}, terminate_on_match=True).get('key') == [[]]:
        self.data['warnings']['linear_dependence'] = True
    if read_pattern(self.text, {'key': '\\*\\*WARNING\\*\\* Hessian does not have the Desired Local Structure'}, terminate_on_match=True).get('key') == [[]]:
        self.data['warnings']['hessian_local_structure'] = True
    if read_pattern(self.text, {'key': '\\*\\*\\*ERROR\\*\\*\\* Exceeded allowed number of iterative cycles in GetCART'}, terminate_on_match=True).get('key') == [[]]:
        self.data['warnings']['GetCART_cycles'] = True
    if read_pattern(self.text, {'key': '\\*\\*WARNING\\*\\* Problems with Internal Coordinates'}, terminate_on_match=True).get('key') == [[]]:
        self.data['warnings']['internal_coordinates'] = True
    if read_pattern(self.text, {'key': 'UNABLE TO DETERMINE Lambda IN RFO  \\*\\*\\s+\\*\\* Taking simple Newton-Raphson step'}, terminate_on_match=True).get('key') == [[]]:
        self.data['warnings']['bad_lambda_take_NR_step'] = True
    if read_pattern(self.text, {'key': 'SWITCHING TO CARTESIAN OPTIMIZATION'}, terminate_on_match=True).get('key') == [[]]:
        self.data['warnings']['switch_to_cartesian'] = True
    if read_pattern(self.text, {'key': '\\*\\*WARNING\\*\\* Magnitude of eigenvalue'}, terminate_on_match=True).get('key') == [[]]:
        self.data['warnings']['eigenvalue_magnitude'] = True
    if read_pattern(self.text, {'key': '\\*\\*WARNING\\*\\* Hereditary positive definiteness endangered'}, terminate_on_match=True).get('key') == [[]]:
        self.data['warnings']['positive_definiteness_endangered'] = True
    if read_pattern(self.text, {'key': '\\*\\*\\*ERROR\\*\\*\\* Angle[\\s\\d]+is near\\-linear\\s+But No atom available to define colinear bend'}, terminate_on_match=True).get('key') == [[]]:
        self.data['warnings']['colinear_bend'] = True
    if read_pattern(self.text, {'key': '\\*\\*\\*ERROR\\*\\*\\* Unable to Diagonalize B\\*B\\(t\\) in <MakeNIC>'}, terminate_on_match=True).get('key') == [[]]:
        self.data['warnings']['diagonalizing_BBt'] = True