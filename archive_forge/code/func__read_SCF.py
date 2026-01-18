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
def _read_SCF(self):
    """Parses both old and new SCFs."""
    if self.data.get('using_GEN_SCFMAN', []):
        footer_pattern = '(^\\s*\\-+\\n\\s+SCF time|^\\s*gen_scfman_exception: SCF failed to converge)'
        header_pattern = '^\\s*\\-+\\s+Cycle\\s+Energy\\s+(?:(?:DIIS)*\\s+[Ee]rror)*(?:RMS Gradient)*\\s+\\-+(?:\\s*\\-+\\s+OpenMP\\s+Integral\\s+computing\\s+Module\\s+(?:Release:\\s+version\\s+[\\d\\-\\.]+\\,\\s+\\w+\\s+[\\d\\-\\.]+\\, Q-Chem Inc\\. Pittsburgh\\s+)*\\-+)*\\n'
        table_pattern = '(?:\\s*Nonlocal correlation = [\\d\\-\\.]+e[\\d\\-]+)*(?:\\s*Inaccurate integrated density:\\n\\s+Number of electrons\\s+=\\s+[\\d\\-\\.]+\\n\\s+Numerical integral\\s+=\\s+[\\d\\-\\.]+\\n\\s+Relative error\\s+=\\s+[\\d\\-\\.]+\\s+\\%\\n)*\\s*\\d+\\s+([\\d\\-\\.]+)\\s+([\\d\\-\\.]+)e([\\d\\-\\.\\+]+)(?:\\s+Convergence criterion met)*(?:\\s+Preconditoned Steepest Descent)*(?:\\s+Roothaan Step)*(?:\\s+(?:Normal\\s+)*BFGS [Ss]tep)*(?:\\s+LineSearch Step)*(?:\\s+Line search: overstep)*(?:\\s+Dog-leg BFGS step)*(?:\\s+Line search: understep)*(?:\\s+Descent step)*(?:\\s+Done DIIS. Switching to GDM)*(?:\\s+Done GDM. Switching to DIIS)*(?:(?:\\s+Done GDM. Switching to GDM with quadratic line-search\\s)*\\s*GDM subspace size\\: \\d+)*(?:\\s*\\-+\\s+Cycle\\s+Energy\\s+(?:(?:DIIS)*\\s+[Ee]rror)*(?:RMS Gradient)*\\s+\\-+(?:\\s*\\-+\\s+OpenMP\\s+Integral\\s+computing\\s+Module\\s+(?:Release:\\s+version\\s+[\\d\\-\\.]+\\,\\s+\\w+\\s+[\\d\\-\\.]+\\, Q-Chem Inc\\. Pittsburgh\\s+)*\\-+)*\\n)*(?:\\s*Line search, dEdstep = [\\d\\-\\.]+e[\\d\\-\\.\\+]+\\s+[\\d\\-\\.]+e[\\d\\-\\.\\+]+\\s+[\\d\\-\\.]+\\s*)*(?:\\s*[\\d\\-\\.]+e[\\d\\-\\.\\+]+\\s+[\\d\\-\\.]+\\s+[\\d\\-\\.]+e[\\d\\-\\.\\+]+\\s+[\\d\\-\\.]+e[\\d\\-\\.\\+]+\\s+[\\d\\-\\.]+\\s+[\\d\\-\\.]+e[\\d\\-\\.\\+]+\\s+Optimal value differs by [\\d\\-\\.]+e[\\d\\-\\.\\+]+ from prediction)*(?:\\s*Resetting GDM\\.)*(?:\\s+[\\d\\-\\.]+e[\\d\\-\\.\\+]+\\s+[\\d\\-\\.]+\\s+[\\d\\-\\.]+e[\\d\\-\\.\\+]+)*(?:\\s+[\\d\\-\\.]+e[\\d\\-\\.\\+]+\\s+[\\d\\-\\.]+\\s+[\\d\\-\\.]+e[\\d\\-\\.\\+]+\\s+[\\d\\-\\.]+e[\\d\\-\\.\\+]+\\s+[\\d\\-\\.]+\\s+[\\d\\-\\.]+e[\\d\\-\\.\\+]+\\s+Optimal value differs by [\\d\\-\\.]+e[\\d\\-\\.\\+]+ from prediction)*(?:\\s*gdm_qls\\: Orbitals will not converge further\\.)*(?:(\\n\\s*[a-z\\dA-Z_\\s/]+\\.C|\\n\\s*GDM)::WARNING energy changes are now smaller than effective accuracy\\.\\s*(\\n\\s*[a-z\\dA-Z_\\s/]+\\.C|\\n\\s*GDM)::\\s+calculation will continue, but THRESH should be increased\\s*(\\n\\s*[a-z\\dA-Z_\\s/]+\\.C|\\n\\s*GDM)::\\s+or SCF_CONVERGENCE decreased\\.\\s*(\\n\\s*[a-z\\dA-Z_\\s/]+\\.C|\\n\\s*GDM)::\\s+effective_thresh = [\\d\\-\\.]+e[\\d\\-]+)*'
    else:
        if 'SCF_failed_to_converge' in self.data.get('errors'):
            footer_pattern = '^\\s*\\d+\\s*[\\d\\-\\.]+\\s+[\\d\\-\\.]+E[\\d\\-\\.]+\\s+Convergence\\s+failure\\n'
        else:
            footer_pattern = '^\\s*\\-+\\n'
        header_pattern = '^\\s*\\-+\\s+Cycle\\s+Energy\\s+DIIS Error\\s+\\-+\\n'
        table_pattern = '(?:\\s*Inaccurate integrated density:\\n\\s+Number of electrons\\s+=\\s+[\\d\\-\\.]+\\n\\s+Numerical integral\\s+=\\s+[\\d\\-\\.]+\\n\\s+Relative error\\s+=\\s+[\\d\\-\\.]+\\s+\\%\\n)*\\s*\\d+\\s*([\\d\\-\\.]+)\\s+([\\d\\-\\.]+)E([\\d\\-\\.\\+]+)(?:\\s*\\n\\s*cpu\\s+[\\d\\-\\.]+\\swall\\s+[\\d\\-\\.]+)*(?:\\nin dftxc\\.C, eleTot sum is:[\\d\\-\\.]+, tauTot is\\:[\\d\\-\\.]+)*(?:\\s+Convergence criterion met)*(?:\\s+Done RCA\\. Switching to DIIS)*(?:\\n\\s*Warning: not using a symmetric Q)*(?:\\nRecomputing EXC\\s*[\\d\\-\\.]+\\s*[\\d\\-\\.]+\\s*[\\d\\-\\.]+(?:\\s*\\nRecomputing EXC\\s*[\\d\\-\\.]+\\s*[\\d\\-\\.]+\\s*[\\d\\-\\.]+)*)*'
    temp_scf = read_table_pattern(self.text, header_pattern, table_pattern, footer_pattern)
    real_scf = []
    for one_scf in temp_scf:
        temp = np.zeros(shape=(len(one_scf), 2))
        for ii, entry in enumerate(one_scf):
            temp[ii, 0] = float(entry[0])
            temp[ii, 1] = float(entry[1]) * 10 ** float(entry[2])
        real_scf += [temp]
    self.data['SCF'] = real_scf
    temp_thresh_warning = read_pattern(self.text, {'key': '\\n[a-zA-Z_\\s/]+\\.C::WARNING energy changes are now smaller than effective accuracy\\.\\n[a-zA-Z_\\s/]+\\.C::\\s+calculation will continue, but THRESH should be increased\\n[a-zA-Z_\\s/]+\\.C::\\s+or SCF_CONVERGENCE decreased\\. \\n[a-zA-Z_\\s/]+\\.C::\\s+effective_thresh = ([\\d\\-\\.]+e[\\d\\-]+)'}).get('key')
    if temp_thresh_warning is not None:
        if len(temp_thresh_warning) == 1:
            self.data['warnings']['thresh'] = float(temp_thresh_warning[0][0])
        else:
            thresh_warning = np.zeros(len(temp_thresh_warning))
            for ii, entry in enumerate(temp_thresh_warning):
                thresh_warning[ii] = float(entry[0])
            self.data['warnings']['thresh'] = thresh_warning
    temp_SCF_energy = read_pattern(self.text, {'key': 'SCF   energy in the final basis set =\\s*([\\d\\-\\.]+)'}).get('key')
    if temp_SCF_energy is not None:
        if len(temp_SCF_energy) == 1:
            self.data['SCF_energy_in_the_final_basis_set'] = float(temp_SCF_energy[0][0])
        else:
            SCF_energy = np.zeros(len(temp_SCF_energy))
            for ii, val in enumerate(temp_SCF_energy):
                SCF_energy[ii] = float(val[0])
            self.data['SCF_energy_in_the_final_basis_set'] = SCF_energy
    temp_Total_energy = read_pattern(self.text, {'key': 'Total energy in the final basis set =\\s*([\\d\\-\\.]+)'}).get('key')
    if temp_Total_energy is not None:
        if len(temp_Total_energy) == 1:
            self.data['Total_energy_in_the_final_basis_set'] = float(temp_Total_energy[0][0])
        else:
            Total_energy = np.zeros(len(temp_Total_energy))
            for ii, val in enumerate(temp_Total_energy):
                Total_energy[ii] = float(val[0])
            self.data['Total_energy_in_the_final_basis_set'] = Total_energy