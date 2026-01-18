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
def _read_frequency_data(self):
    """Parses cpscf_nseg, frequencies, enthalpy, entropy, and mode vectors."""
    if read_pattern(self.text, {'key': 'Calculating MO derivatives via CPSCF'}, terminate_on_match=True).get('key') == [[]]:
        temp_cpscf_nseg = read_pattern(self.text, {'key': 'CPSCF will be done in([\\d\\s]+)segments to save memory'}, terminate_on_match=True).get('key')
        if temp_cpscf_nseg is None:
            self.data['cpscf_nseg'] = 1
        else:
            self.data['cpscf_nseg'] = int(temp_cpscf_nseg[0][0])
    else:
        self.data['cpscf_nseg'] = 0
    raman = False
    if read_pattern(self.text, {'key': 'doraman\\s*(?:=)*\\s*true'}, terminate_on_match=True).get('key') == [[]]:
        raman = True
    temp_dict = read_pattern(self.text, {'frequencies': '\\s*Frequency:\\s+(\\-?[\\d\\.\\*]+)(?:\\s+(\\-?[\\d\\.\\*]+)(?:\\s+(\\-?[\\d\\.\\*]+))*)*', 'trans_dip': 'TransDip\\s+(\\-?[\\d\\.]{5,7}|\\*{5,7})\\s*(\\-?[\\d\\.]{5,7}|\\*{5,7})\\s*(\\-?[\\d\\.]{5,7}|\\*{5,7})\\s*(?:(\\-?[\\d\\.]{5,7}|\\*{5,7})\\s*(\\-?[\\d\\.]{5,7}|\\*{5,7})\\s*(\\-?[\\d\\.]{5,7}|\\*{5,7})\\s*(?:(\\-?[\\d\\.]{5,7}|\\*{5,7})\\s*(\\-?[\\d\\.]{5,7}|\\*{5,7})\\s*(\\-?[\\d\\.]{5,7}|\\*{5,7}))*)*', 'IR_intens': '\\s*IR Intens:\\s*(\\-?[\\d\\.\\*]+)(?:\\s+(\\-?[\\d\\.\\*]+)(?:\\s+(\\-?[\\d\\.\\*]+))*)*', 'IR_active': '\\s*IR Active:\\s+([YESNO]+)(?:\\s+([YESNO]+)(?:\\s+([YESNO]+))*)*', 'raman_intens': '\\s*Raman Intens:\\s*(\\-?[\\d\\.\\*]+)(?:\\s+(\\-?[\\d\\.\\*]+)(?:\\s+(\\-?[\\d\\.\\*]+))*)*', 'depolar': '\\s*Depolar:\\s*(\\-?[\\d\\.\\*]+)(?:\\s+(\\-?[\\d\\.\\*]+)(?:\\s+(\\-?[\\d\\.\\*]+))*)*', 'raman_active': '\\s*Raman Active:\\s+([YESNO]+)(?:\\s+([YESNO]+)(?:\\s+([YESNO]+))*)*', 'ZPE': '\\s*Zero point vibrational energy:\\s+([\\d\\-\\.]+)\\s+kcal/mol', 'trans_enthalpy': '\\s*Translational Enthalpy:\\s+([\\d\\-\\.]+)\\s+kcal/mol', 'rot_enthalpy': '\\s*Rotational Enthalpy:\\s+([\\d\\-\\.]+)\\s+kcal/mol', 'vib_enthalpy': '\\s*Vibrational Enthalpy:\\s+([\\d\\-\\.]+)\\s+kcal/mol', 'gas_constant': '\\s*gas constant \\(RT\\):\\s+([\\d\\-\\.]+)\\s+kcal/mol', 'trans_entropy': '\\s*Translational Entropy:\\s+([\\d\\-\\.]+)\\s+cal/mol\\.K', 'rot_entropy': '\\s*Rotational Entropy:\\s+([\\d\\-\\.]+)\\s+cal/mol\\.K', 'vib_entropy': '\\s*Vibrational Entropy:\\s+([\\d\\-\\.]+)\\s+cal/mol\\.K', 'total_enthalpy': '\\s*Total Enthalpy:\\s+([\\d\\-\\.]+)\\s+kcal/mol', 'total_entropy': '\\s*Total Entropy:\\s+([\\d\\-\\.]+)\\s+cal/mol\\.K'})
    keys = ['ZPE', 'trans_enthalpy', 'rot_enthalpy', 'vib_enthalpy', 'gas_constant', 'trans_entropy', 'rot_entropy', 'vib_entropy', 'total_enthalpy', 'total_entropy']
    for key in keys:
        if temp_dict.get(key) is None:
            self.data[key] = None
        else:
            self.data[key] = float(temp_dict.get(key)[0][0])
    if temp_dict.get('frequencies') is None:
        self.data['frequencies'] = self.data['IR_intens'] = self.data['IR_active'] = None
        self.data['raman_active'] = self.data['raman_intens'] = None
        self.data['depolar'] = self.data['trans_dip'] = None
    else:
        temp_freqs = [value for entry in temp_dict.get('frequencies') for value in entry]
        temp_IR_intens = [value for entry in temp_dict.get('IR_intens') for value in entry]
        IR_active = [value for entry in temp_dict.get('IR_active') for value in entry]
        temp_trans_dip = [value for entry in temp_dict.get('trans_dip') for value in entry]
        self.data['IR_active'] = IR_active
        if raman:
            raman_active = [value for entry in temp_dict.get('raman_active') for value in entry]
            temp_raman_intens = [value for entry in temp_dict.get('raman_intens') for value in entry]
            temp_depolar = [value for entry in temp_dict.get('depolar') for value in entry]
            self.data['raman_active'] = raman_active
            raman_intens = np.zeros(len(temp_raman_intens) - temp_raman_intens.count('None'))
            for ii, entry in enumerate(temp_raman_intens):
                if entry != 'None':
                    if '*' in entry:
                        raman_intens[ii] = float('inf')
                    else:
                        raman_intens[ii] = float(entry)
            self.data['raman_intens'] = raman_intens
            depolar = np.zeros(len(temp_depolar) - temp_depolar.count('None'))
            for ii, entry in enumerate(temp_depolar):
                if entry != 'None':
                    if '*' in entry:
                        depolar[ii] = float('inf')
                    else:
                        depolar[ii] = float(entry)
            self.data['depolar'] = depolar
        else:
            self.data['raman_intens'] = self.data['raman_active'] = self.data['depolar'] = None
        trans_dip = np.zeros(shape=(int((len(temp_trans_dip) - temp_trans_dip.count('None')) / 3), 3))
        for ii, entry in enumerate(temp_trans_dip):
            if entry != 'None':
                if '*' in entry:
                    trans_dip[int(ii / 3)][ii % 3] = float('inf')
                else:
                    trans_dip[int(ii / 3)][ii % 3] = float(entry)
        self.data['trans_dip'] = trans_dip
        freqs = np.zeros(len(temp_freqs) - temp_freqs.count('None'))
        for ii, entry in enumerate(temp_freqs):
            if entry != 'None':
                if '*' in entry:
                    if ii == 0:
                        freqs[ii] = -float('inf')
                    elif ii == len(freqs) - 1:
                        freqs[ii] = float('inf')
                    elif freqs[ii - 1] == -float('inf'):
                        freqs[ii] = -float('inf')
                    elif '*' in temp_freqs[ii + 1]:
                        freqs[ii] = float('inf')
                    else:
                        raise RuntimeError('ERROR: Encountered an undefined frequency not at the beginning or end of the frequency list, which makes no sense! Exiting...')
                    if not self.data.get('completion', []):
                        if 'undefined_frequency' not in self.data['errors']:
                            self.data['errors'] += ['undefined_frequency']
                    elif 'undefined_frequency' not in self.data['warnings']:
                        self.data['warnings']['undefined_frequency'] = True
                else:
                    freqs[ii] = float(entry)
        self.data['frequencies'] = freqs
        IR_intens = np.zeros(len(temp_IR_intens) - temp_IR_intens.count('None'))
        for ii, entry in enumerate(temp_IR_intens):
            if entry != 'None':
                if '*' in entry:
                    IR_intens[ii] = float('inf')
                else:
                    IR_intens[ii] = float(entry)
        self.data['IR_intens'] = IR_intens
        if not raman:
            header_pattern = '\\s*Raman Active:\\s+[YESNO]+\\s+(?:[YESNO]+\\s+)*X\\s+Y\\s+Z\\s+(?:X\\s+Y\\s+Z\\s+)*'
        else:
            header_pattern = '\\s*Depolar:\\s*\\-?[\\d\\.\\*]+\\s+(?:\\-?[\\d\\.\\*]+\\s+)*X\\s+Y\\s+Z\\s+(?:X\\s+Y\\s+Z\\s+)*'
        table_pattern = '\\s*[a-zA-Z][a-zA-Z\\s]\\s*([\\d\\-\\.]+)\\s*([\\d\\-\\.]+)\\s*([\\d\\-\\.]+)\\s*(?:([\\d\\-\\.]+)\\s*([\\d\\-\\.]+)\\s*([\\d\\-\\.]+)\\s*(?:([\\d\\-\\.]+)\\s*([\\d\\-\\.]+)\\s*([\\d\\-\\.]+))*)*'
        footer_pattern = 'TransDip\\s+\\-?[\\d\\.\\*]+\\s*\\-?[\\d\\.\\*]+\\s*\\-?[\\d\\.\\*]+\\s*(?:\\-?[\\d\\.\\*]+\\s*\\-?[\\d\\.\\*]+\\s*\\-?[\\d\\.\\*]+\\s*)*'
        temp_freq_mode_vecs = read_table_pattern(self.text, header_pattern, table_pattern, footer_pattern)
        freq_mode_vecs = np.zeros(shape=(len(freqs), len(temp_freq_mode_vecs[0]), 3))
        for ii, triple_FMV in enumerate(temp_freq_mode_vecs):
            for jj, line in enumerate(triple_FMV):
                for kk, entry in enumerate(line):
                    if entry != 'None':
                        freq_mode_vecs[int(ii * 3 + math.floor(kk / 3)), jj, kk % 3] = float(entry)
        self.data['frequency_mode_vectors'] = freq_mode_vecs
        freq_length = len(self.data['frequencies'])
        if len(self.data['frequency_mode_vectors']) != freq_length or len(self.data['IR_intens']) != freq_length or len(self.data['IR_active']) != freq_length:
            self.data['warnings']['frequency_length_inconsistency'] = True