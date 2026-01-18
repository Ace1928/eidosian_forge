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
def _read_charges_and_dipoles(self):
    """
        Parses Mulliken/ESP/RESP charges.
        Parses associated dipole/multipole moments.
        Also parses spins given an unrestricted SCF.
        """
    self.data['dipoles'] = {}
    temp_dipole_total = read_pattern(self.text, {'key': 'X\\s*[\\d\\-\\.]+\\s*Y\\s*[\\d\\-\\.]+\\s*Z\\s*[\\d\\-\\.]+\\s*Tot\\s*([\\d\\-\\.]+)'}).get('key')
    temp_dipole = read_pattern(self.text, {'key': 'X\\s*([\\d\\-\\.]+)\\s*Y\\s*([\\d\\-\\.]+)\\s*Z\\s*([\\d\\-\\.]+)\\s*Tot\\s*[\\d\\-\\.]+'}).get('key')
    if temp_dipole is not None:
        if len(temp_dipole_total) == 1:
            self.data['dipoles']['total'] = float(temp_dipole_total[0][0])
            dipole = np.zeros(3)
            for ii, val in enumerate(temp_dipole[0]):
                dipole[ii] = float(val)
            self.data['dipoles']['dipole'] = dipole
        else:
            total = np.zeros(len(temp_dipole_total))
            for ii, val in enumerate(temp_dipole_total):
                total[ii] = float(val[0])
            self.data['dipoles']['total'] = total
            dipole = np.zeros(shape=(len(temp_dipole_total), 3))
            for ii in range(len(temp_dipole)):
                for jj, _val in enumerate(temp_dipole[ii]):
                    dipole[ii][jj] = temp_dipole[ii][jj]
            self.data['dipoles']['dipole'] = dipole
    self.data['multipoles'] = dict()
    quad_mom_pat = '\\s*Quadrupole Moments \\(Debye\\-Ang\\)\\s+XX\\s+([\\-\\.0-9]+)\\s+XY\\s+([\\-\\.0-9]+)\\s+YY\\s+([\\-\\.0-9]+)\\s+XZ\\s+([\\-\\.0-9]+)\\s+YZ\\s+([\\-\\.0-9]+)\\s+ZZ\\s+([\\-\\.0-9]+)'
    temp_quadrupole_moment = read_pattern(self.text, {'key': quad_mom_pat}).get('key')
    if temp_quadrupole_moment is not None:
        keys = ('XX', 'XY', 'YY', 'XZ', 'YZ', 'ZZ')
        if len(temp_quadrupole_moment) == 1:
            self.data['multipoles']['quadrupole'] = {key: float(temp_quadrupole_moment[0][idx]) for idx, key in enumerate(keys)}
        else:
            self.data['multipoles']['quadrupole'] = list()
            for qpole in temp_quadrupole_moment:
                self.data['multipoles']['quadrupole'].append({key: float(qpole[idx]) for idx, key in enumerate(keys)})
    octo_mom_pat = '\\s*Octopole Moments \\(Debye\\-Ang\\^2\\)\\s+XXX\\s+([\\-\\.0-9]+)\\s+XXY\\s+([\\-\\.0-9]+)\\s+XYY\\s+([\\-\\.0-9]+)\\s+YYY\\s+([\\-\\.0-9]+)\\s+XXZ\\s+([\\-\\.0-9]+)\\s+XYZ\\s+([\\-\\.0-9]+)\\s+YYZ\\s+([\\-\\.0-9]+)\\s+XZZ\\s+([\\-\\.0-9]+)\\s+YZZ\\s+([\\-\\.0-9]+)\\s+ZZZ\\s+([\\-\\.0-9]+)'
    temp_octopole_moment = read_pattern(self.text, {'key': octo_mom_pat}).get('key')
    if temp_octopole_moment is not None:
        keys = ('XXX', 'XXY', 'XYY', 'YYY', 'XXZ', 'XYZ', 'YYZ', 'XZZ', 'YZZ', 'ZZZ')
        if len(temp_octopole_moment) == 1:
            self.data['multipoles']['octopole'] = {key: float(temp_octopole_moment[0][idx]) for idx, key in enumerate(keys)}
        else:
            self.data['multipoles']['octopole'] = list()
            for opole in temp_octopole_moment:
                self.data['multipoles']['octopole'].append({key: float(opole[idx]) for idx, key in enumerate(keys)})
    hexadeca_mom_pat = '\\s*Hexadecapole Moments \\(Debye\\-Ang\\^3\\)\\s+XXXX\\s+([\\-\\.0-9]+)\\s+XXXY\\s+([\\-\\.0-9]+)\\s+XXYY\\s+([\\-\\.0-9]+)\\s+XYYY\\s+([\\-\\.0-9]+)\\s+YYYY\\s+([\\-\\.0-9]+)\\s+XXXZ\\s+([\\-\\.0-9]+)\\s+XXYZ\\s+([\\-\\.0-9]+)\\s+XYYZ\\s+([\\-\\.0-9]+)\\s+YYYZ\\s+([\\-\\.0-9]+)\\s+XXZZ\\s+([\\-\\.0-9]+)\\s+XYZZ\\s+([\\-\\.0-9]+)\\s+YYZZ\\s+([\\-\\.0-9]+)\\s+XZZZ\\s+([\\-\\.0-9]+)\\s+YZZZ\\s+([\\-\\.0-9]+)\\s+ZZZZ\\s+([\\-\\.0-9]+)'
    temp_hexadecapole_moment = read_pattern(self.text, {'key': hexadeca_mom_pat}).get('key')
    if temp_hexadecapole_moment is not None:
        keys = ('XXXX', 'XXXY', 'XXYY', 'XYYY', 'YYYY', 'XXXZ', 'XXYZ', 'XYYZ', 'YYYZ', 'XXZZ', 'XYZZ', 'YYZZ', 'XZZZ', 'YZZZ', 'ZZZZ')
        if len(temp_hexadecapole_moment) == 1:
            self.data['multipoles']['hexadecapole'] = {key: float(temp_hexadecapole_moment[0][idx]) for idx, key in enumerate(keys)}
        else:
            self.data['multipoles']['hexadecapole'] = list()
            for hpole in temp_hexadecapole_moment:
                self.data['multipoles']['hexadecapole'].append({key: float(hpole[idx]) for idx, key in enumerate(keys)})
    if self.data.get('unrestricted', []):
        header_pattern = '\\-+\\s+Ground-State Mulliken Net Atomic Charges\\s+Atom\\s+Charge \\(a\\.u\\.\\)\\s+Spin\\s\\(a\\.u\\.\\)\\s+\\-+'
        table_pattern = '\\s+\\d+\\s\\w+\\s+([\\d\\-\\.]+)\\s+([\\d\\-\\.]+)'
        footer_pattern = '\\s\\s\\-+\\s+Sum of atomic charges'
    else:
        header_pattern = '\\-+\\s+Ground-State Mulliken Net Atomic Charges\\s+Atom\\s+Charge \\(a\\.u\\.\\)\\s+\\-+'
        table_pattern = '\\s+\\d+\\s\\w+\\s+([\\d\\-\\.]+)'
        footer_pattern = '\\s\\s\\-+\\s+Sum of atomic charges'
    temp_mulliken = read_table_pattern(self.text, header_pattern, table_pattern, footer_pattern)
    real_mulliken = []
    for one_mulliken in temp_mulliken:
        if self.data.get('unrestricted', []):
            temp = np.zeros(shape=(len(one_mulliken), 2))
            for ii, entry in enumerate(one_mulliken):
                temp[ii, 0] = float(entry[0])
                temp[ii, 1] = float(entry[1])
        else:
            temp = np.zeros(len(one_mulliken))
            for ii, entry in enumerate(one_mulliken):
                temp[ii] = float(entry[0])
        real_mulliken += [temp]
    self.data['Mulliken'] = real_mulliken
    esp_or_resp = read_pattern(self.text, {'key': 'Merz-Kollman (R?ESP) Net Atomic Charges'}).get('key')
    if esp_or_resp is not None:
        header_pattern = 'Merz-Kollman (R?ESP) Net Atomic Charges\\s+Atom\\s+Charge \\(a\\.u\\.\\)\\s+\\-+'
        table_pattern = '\\s+\\d+\\s\\w+\\s+([\\d\\-\\.]+)'
        footer_pattern = '\\s\\s\\-+\\s+Sum of atomic charges'
        temp_esp_or_resp = read_table_pattern(self.text, header_pattern, table_pattern, footer_pattern)
        real_esp_or_resp = []
        for one_entry in temp_esp_or_resp:
            temp = np.zeros(len(one_entry))
            for ii, entry in enumerate(one_entry):
                temp[ii] = float(entry[0])
            real_esp_or_resp += [temp]
        self.data[esp_or_resp[0][0]] = real_esp_or_resp
        temp_RESP_dipole_total = read_pattern(self.text, {'key': 'Related Dipole Moment =\\s*([\\d\\-\\.]+)\\s*\\(X\\s*[\\d\\-\\.]+\\s*Y\\s*[\\d\\-\\.]+\\s*Z\\s*[\\d\\-\\.]+\\)'}).get('key')
        temp_RESP_dipole = read_pattern(self.text, {'key': 'Related Dipole Moment =\\s*[\\d\\-\\.]+\\s*\\(X\\s*([\\d\\-\\.]+)\\s*Y\\s*([\\d\\-\\.]+)\\s*Z\\s*([\\d\\-\\.]+)\\)'}).get('key')
        if temp_RESP_dipole is not None:
            if len(temp_RESP_dipole_total) == 1:
                self.data['dipoles']['RESP_total'] = float(temp_RESP_dipole_total[0][0])
                RESP_dipole = np.zeros(3)
                for ii, val in enumerate(temp_RESP_dipole[0]):
                    RESP_dipole[ii] = float(val)
                self.data['dipoles']['RESP_dipole'] = RESP_dipole
            else:
                RESP_total = np.zeros(len(temp_RESP_dipole_total))
                for ii, val in enumerate(temp_RESP_dipole_total):
                    RESP_total[ii] = float(val[0])
                self.data['dipoles']['RESP_total'] = RESP_total
                RESP_dipole = np.zeros(shape=(len(temp_RESP_dipole_total), 3))
                for ii in range(len(temp_RESP_dipole)):
                    for jj, _val in enumerate(temp_RESP_dipole[ii]):
                        RESP_dipole[ii][jj] = temp_RESP_dipole[ii][jj]
                self.data['dipoles']['RESP_dipole'] = RESP_dipole