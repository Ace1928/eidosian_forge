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
def _check_completion_errors(self):
    """Parses potential errors that can cause jobs to crash."""
    if read_pattern(self.text, {'key': 'Coordinates do not transform within specified threshold'}, terminate_on_match=True).get('key') == [[]]:
        self.data['errors'] += ['failed_to_transform_coords']
    elif read_pattern(self.text, {'key': 'The Q\\-Chem input file has failed to pass inspection'}, terminate_on_match=True).get('key') == [[]]:
        self.data['errors'] += ['input_file_error']
    elif read_pattern(self.text, {'key': 'Error opening input stream'}, terminate_on_match=True).get('key') == [[]]:
        self.data['errors'] += ['failed_to_read_input']
    elif read_pattern(self.text, {'key': 'FileMan error: End of file reached prematurely'}, terminate_on_match=True).get('key') == [[]]:
        self.data['errors'] += ['premature_end_FileMan_error']
    elif read_pattern(self.text, {'key': 'need to increase the array of NLebdevPts'}, terminate_on_match=True).get('key') == [[]]:
        self.data['errors'] += ['NLebdevPts']
    elif read_pattern(self.text, {'key': 'method not available'}, terminate_on_match=True).get('key') == [[]]:
        self.data['errors'] += ['method_not_available']
    elif read_pattern(self.text, {'key': 'Could not find \\$molecule section in ParseQInput'}, terminate_on_match=True).get('key') == [[]]:
        self.data['errors'] += ['read_molecule_error']
    elif read_pattern(self.text, {'key': 'Welcome to Q-Chem'}, terminate_on_match=True).get('key') != [[]]:
        self.data['errors'] += ['never_called_qchem']
    elif read_pattern(self.text, {'key': '\\*\\*\\*ERROR\\*\\*\\* Hessian Appears to have all zero or negative eigenvalues'}, terminate_on_match=True).get('key') == [[]]:
        self.data['errors'] += ['hessian_eigenvalue_error']
    elif read_pattern(self.text, {'key': 'FlexNet Licensing error'}, terminate_on_match=True).get('key') == [[]] or read_pattern(self.text, {'key': 'Unable to validate license'}, terminate_on_match=True).get('key') == [[]]:
        self.data['errors'] += ['licensing_error']
    elif read_pattern(self.text, {'key': 'Could not open driver file in ReadDriverFromDisk'}, terminate_on_match=True).get('key') == [[]]:
        self.data['errors'] += ['driver_error']
    elif read_pattern(self.text, {'key': 'Basis not supported for the above atom'}, terminate_on_match=True).get('key') == [[]]:
        self.data['errors'] += ['basis_not_supported']
    elif read_pattern(self.text, {'key': 'Unable to find relaxed density'}, terminate_on_match=True).get('key') == [[]] or read_pattern(self.text, {'key': 'Out of Iterations- IterZ'}, terminate_on_match=True).get('key') == [[]]:
        self.data['errors'] += ['failed_cpscf']
    elif read_pattern(self.text, {'key': 'RUN_NBO6 \\(rem variable\\) is not correct'}, terminate_on_match=True).get('key') == [[]]:
        self.data['errors'] += ['bad_old_nbo6_rem']
    elif read_pattern(self.text, {'key': 'NBO_EXTERNAL \\(rem variable\\) is not correct'}, terminate_on_match=True).get('key') == [[]]:
        self.data['errors'] += ['bad_new_nbo_external_rem']
    elif read_pattern(self.text, {'key': 'gen_scfman_exception:  GDM:: Zero or negative preconditioner scaling factor'}, terminate_on_match=True).get('key') == [[]]:
        self.data['errors'] += ['gdm_neg_precon_error']
    elif read_pattern(self.text, {'key': 'too many atoms in ESPChgFit'}, terminate_on_match=True).get('key') == [[]]:
        self.data['errors'] += ['esp_chg_fit_error']
    elif read_pattern(self.text, {'key': 'Please use larger MEM_STATIC'}, terminate_on_match=True).get('key') == [[]] or read_pattern(self.text, {'key': 'Please increase MEM_STATIC'}, terminate_on_match=True).get('key') == [[]]:
        self.data['errors'] += ['mem_static_too_small']
    elif read_pattern(self.text, {'key': 'Please increase MEM_TOTAL'}, terminate_on_match=True).get('key') == [[]]:
        self.data['errors'] += ['mem_total_too_small']
    elif self.text[-34:-2] == 'Computing fast CPCM-SWIG hessian' or self.text[-16:-1] == 'Roots Converged':
        self.data['errors'] += ['probably_out_of_memory']
    else:
        tmp_failed_line_searches = read_pattern(self.text, {'key': '\\d+\\s+failed line searches\\.\\s+Resetting'}, terminate_on_match=False).get('key')
        if tmp_failed_line_searches is not None and len(tmp_failed_line_searches) > 10:
            self.data['errors'] += ['SCF_failed_to_converge']
    if self.data.get('errors') == []:
        self.data['errors'] += ['unknown_error']