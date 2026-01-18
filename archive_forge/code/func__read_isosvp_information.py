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
def _read_isosvp_information(self):
    """
        Parses information from ISOSVP solvent calculations.

        There are 5 energies output, as in the example below

        --------------------------------------------------------------------------------
        The Final SS(V)PE energies and Properties
        --------------------------------------------------------------------------------

        Energies
        --------------------
        The Final Solution-Phase Energy =     -40.4850599390
        The Solute Internal Energy =          -40.4846329759
        The Change in Solute Internal Energy =  0.0000121970  (   0.00765 KCAL/MOL)
        The Reaction Field Free Energy =       -0.0004269631  (  -0.26792 KCAL/MOL)
        The Total Solvation Free Energy =      -0.0004147661  (  -0.26027 KCAL/MOL)

        In addition, we need to parse the DIELST fortran variable to get the dielectric
        constant used.
        """
    temp_dict = read_pattern(self.text, {'final_soln_phase_e': '\\s*The Final Solution-Phase Energy\\s+=\\s+([\\d\\-\\.]+)\\s*', 'solute_internal_e': '\\s*The Solute Internal Energy\\s+=\\s+([\\d\\-\\.]+)\\s*', 'total_solvation_free_e': '\\s*The Total Solvation Free Energy\\s+=\\s+([\\d\\-\\.]+)\\s*', 'change_solute_internal_e': '\\s*The Change in Solute Internal Energy\\s+=\\s+(\\s+[\\d\\-\\.]+)\\s+\\(\\s+([\\d\\-\\.]+)\\s+KCAL/MOL\\)\\s*', 'reaction_field_free_e': '\\s*The Reaction Field Free Energy\\s+=\\s+(\\s+[\\d\\-\\.]+)\\s+\\(\\s+([\\d\\-\\.]+)\\s+KCAL/MOL\\)\\s*', 'isosvp_dielectric': '\\s*DIELST=\\s+(\\s+[\\d\\-\\.]+)\\s*'})
    for key in temp_dict:
        if temp_dict.get(key) is None:
            self.data['solvent_data']['isosvp'][key] = None
        elif len(temp_dict.get(key)) == 1:
            self.data['solvent_data']['isosvp'][key] = float(temp_dict.get(key)[0][0])
        else:
            temp_result = np.zeros(len(temp_dict.get(key)))
            for ii, entry in enumerate(temp_dict.get(key)):
                temp_result[ii] = float(entry[0])
            self.data['solvent_data']['isosvp'][key] = temp_result