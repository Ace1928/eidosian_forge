from __future__ import annotations
import os
import re
from typing import TYPE_CHECKING
from monty.io import reverse_readline
from monty.itertools import chunks
from monty.json import MSONable
from monty.serialization import zopen
from pymatgen.core.structure import Molecule
def _parse_adf_output(self):
    """Parse the standard ADF output file."""
    numerical_freq_patt = re.compile('\\s+\\*\\s+F\\sR\\sE\\sQ\\sU\\sE\\sN\\sC\\sI\\sE\\sS\\s+\\*')
    analytic_freq_patt = re.compile('\\s+\\*\\s+F\\sR\\sE\\sQ\\sU\\sE\\sN\\sC\\sY\\s+A\\sN\\sA\\sL\\sY\\sS\\sI\\sS\\s+\\*')
    freq_on_patt = re.compile('Vibrations\\sand\\sNormal\\sModes\\s+\\*+.*\\*+')
    freq_off_patt = re.compile('List\\sof\\sAll\\sFrequencies:')
    mode_patt = re.compile('\\s+(\\d+)\\.([A-Za-z]+)\\s+(.*)')
    coord_patt = re.compile('\\s+(\\d+)\\s+([A-Za-z]+)' + 6 * '\\s+([0-9\\.-]+)')
    coord_on_patt = re.compile('\\s+\\*\\s+R\\sU\\sN\\s+T\\sY\\sP\\sE\\s:\\sFREQUENCIES\\s+\\*')
    parse_freq = False
    parse_mode = False
    n_next = 0
    n_strike = 0
    sites = []
    self.frequencies = []
    self.normal_modes = []
    if self.final_structure is None:
        find_structure = True
        parse_coord = False
        n_atoms = 0
    else:
        find_structure = False
        parse_coord = False
        n_atoms = len(self.final_structure)
    with open(self.filename) as file:
        for line in file:
            if self.run_type == 'NumericalFreq' and find_structure:
                if not parse_coord:
                    m = coord_on_patt.search(line)
                    if m:
                        parse_coord = True
                else:
                    m = coord_patt.search(line)
                    if m:
                        sites.append([m.group(2), list(map(float, m.groups()[2:5]))])
                        n_strike += 1
                    elif n_strike > 0:
                        find_structure = False
                        self.final_structure = self._sites_to_mol(sites)
                        n_atoms = len(self.final_structure)
            elif self.freq_type is None:
                if numerical_freq_patt.search(line):
                    self.freq_type = 'Numerical'
                elif analytic_freq_patt.search(line):
                    self.freq_type = 'Analytical'
                    self.run_type = 'AnalyticalFreq'
            elif freq_on_patt.search(line):
                parse_freq = True
            elif parse_freq:
                if freq_off_patt.search(line):
                    break
                el = line.strip().split()
                if 1 <= len(el) <= 3 and line.find('.') != -1:
                    n_next = len(el)
                    parse_mode = True
                    parse_freq = False
                    self.frequencies.extend(map(float, el))
                    for _ in range(n_next):
                        self.normal_modes.append([])
            elif parse_mode:
                m = mode_patt.search(line)
                if m:
                    v = list(chunks(map(float, m.group(3).split()), 3))
                    if len(v) != n_next:
                        raise AdfOutputError('Odd Error!')
                    for i, k in enumerate(range(-n_next, 0)):
                        self.normal_modes[k].extend(v[i])
                    if int(m.group(1)) == n_atoms:
                        parse_freq = True
                        parse_mode = False
    if isinstance(self.final_structure, list):
        self.final_structure = self._sites_to_mol(self.final_structure)
    if self.freq_type is not None:
        if len(self.frequencies) != len(self.normal_modes):
            raise AdfOutputError('The number of normal modes is wrong!')
        if len(self.normal_modes[0]) != n_atoms * 3:
            raise AdfOutputError('The dimensions of the modes are wrong!')