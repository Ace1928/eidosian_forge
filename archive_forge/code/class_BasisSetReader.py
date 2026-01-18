from __future__ import annotations
import os
import re
import shutil
import subprocess
from string import Template
from typing import TYPE_CHECKING
from monty.io import zopen
from monty.json import MSONable
from pymatgen.core.structure import Molecule
class BasisSetReader:
    """
    A basis set reader.
    Basis set are stored in data as a dict:
    :key l_zeta_ng for each nl orbitals which contain list of tuple (alpha, coef) for each of the ng gaussians
    in l_zeta orbital.
    """

    def __init__(self, filename):
        """
        Args:
            filename: Filename to read.
        """
        self.filename = filename
        with zopen(filename) as file:
            basis_set = file.read()
        self.data = self._parse_file(basis_set)
        self.data.update(n_nlmo=self.set_n_nlmo())

    @staticmethod
    def _parse_file(input):
        lmax_nnlo_patt = re.compile('\\s* (\\d+) \\s+ (\\d+) \\s+ \\# .* ', re.VERBOSE)
        nl_orbital_patt = re.compile('\\s* (\\d+) \\s+ (\\d+) \\s+ (\\d+) \\s+ \\# .* ', re.VERBOSE)
        coef_alpha_patt = re.compile('\\s* ([-\\d.\\D]+) \\s+ ([-\\d.\\D]+) \\s* ', re.VERBOSE)
        preamble = []
        basis_set = {}
        parse_preamble = False
        parse_lmax_nnlo = False
        parse_nl_orbital = False
        nnlo = None
        lmax = None
        for line in input.split('\n'):
            if parse_nl_orbital:
                match_orb = nl_orbital_patt.search(line)
                match_alpha = coef_alpha_patt.search(line)
                if match_orb:
                    l_angular = match_orb.group(1)
                    zeta = match_orb.group(2)
                    ng = match_orb.group(3)
                    basis_set[f'{l_angular}_{zeta}_{ng}'] = []
                elif match_alpha:
                    alpha = match_alpha.group(1)
                    coef = match_alpha.group(2)
                    basis_set[f'{l_angular}_{zeta}_{ng}'].append((alpha, coef))
            elif parse_lmax_nnlo:
                match_orb = lmax_nnlo_patt.search(line)
                if match_orb:
                    lmax = match_orb.group(1)
                    nnlo = match_orb.group(2)
                    parse_lmax_nnlo = False
                    parse_nl_orbital = True
            elif parse_preamble:
                preamble.append(line.strip())
            if line.find('</preamble>') != -1:
                parse_preamble = False
                parse_lmax_nnlo = True
            elif line.find('<preamble>') != -1:
                parse_preamble = True
        basis_set.update(lmax=lmax, n_nlo=nnlo, preamble=preamble)
        return basis_set

    def set_n_nlmo(self):
        """the number of nlm orbitals for the basis set"""
        n_nlm_orbs = 0
        data_tmp = self.data
        data_tmp.pop('lmax')
        data_tmp.pop('n_nlo')
        data_tmp.pop('preamble')
        for l_zeta_ng in data_tmp:
            n_l = l_zeta_ng.split('_')[0]
            n_nlm_orbs = n_nlm_orbs + (2 * int(n_l) + 1)
        return str(n_nlm_orbs)

    def infos_on_basis_set(self):
        return f'=========================================\nReading basis set:\n\nBasis set for {self.filename} atom \nMaximum angular momentum = {self.data['lmax']}\nNumber of atomics orbitals = {self.data['n_nlo']}\nNumber of nlm orbitals = {self.data['n_nlmo']}\n========================================='