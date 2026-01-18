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
class FiestaOutput:
    """
    A Fiesta output file parser.

    All energies are in eV.
    """

    def __init__(self, filename):
        """
        Args:
            filename: Filename to read.
        """
        self.filename = filename
        with zopen(filename) as file:
            data = file.read()
        chunks = re.split('GW Driver iteration', data)
        chunks.pop(0)
        self.data = [self._parse_job(c) for c in chunks]

    @staticmethod
    def _parse_job(output):
        GW_BANDS_results_patt = re.compile('^<it.*  \\| \\s+ (\\D+\\d*) \\s+ \\| \\s+ ([-\\d.]+) \\s+ ([-\\d.]+) \\s+ ([-\\d.]+) \\s+ \\|  \\s+ ([-\\d.]+) \\s+ ([-\\d.]+) \\s+ ([-\\d.]+) \\s+ \\| \\s+ ([-\\d.]+) \\s+ ([-\\d.]+) \\s+ ', re.VERBOSE)
        GW_GAPS_results_patt = re.compile('^<it.*  \\| \\s+ Egap_KS \\s+ = \\s+ ([-\\d.]+) \\s+ \\| \\s+ Egap_QP \\s+ = \\s+ ([-\\d.]+) \\s+ \\|  \\s+ Egap_QP \\s+ = \\s+ ([-\\d.]+) \\s+ \\|', re.VERBOSE)
        end_patt = re.compile('\\s*program returned normally\\s*')
        total_time_patt = re.compile('\\s*total \\s+ time: \\s+  ([\\d.]+) .*', re.VERBOSE)
        GW_results = {}
        parse_gw_results = False
        parse_total_time = False
        for line in output.split('\n'):
            if parse_total_time:
                m = end_patt.search(line)
                if m:
                    GW_results.update(end_normally=True)
                m = total_time_patt.search(line)
                if m:
                    GW_results.update(total_time=m.group(1))
            if parse_gw_results:
                if line.find('Dumping eigen energies') != -1:
                    parse_total_time = True
                    parse_gw_results = False
                    continue
                m = GW_BANDS_results_patt.search(line)
                if m:
                    dct = {}
                    dct.update(band=m.group(1).strip(), eKS=m.group(2), eXX=m.group(3), eQP_old=m.group(4), z=m.group(5), sigma_c_Linear=m.group(6), eQP_Linear=m.group(7), sigma_c_SCF=m.group(8), eQP_SCF=m.group(9))
                    GW_results[m.group(1).strip()] = dct
                n = GW_GAPS_results_patt.search(line)
                if n:
                    dct = {}
                    dct.update(Egap_KS=n.group(1), Egap_QP_Linear=n.group(2), Egap_QP_SCF=n.group(3))
                    GW_results['Gaps'] = dct
            if line.find('GW Results') != -1:
                parse_gw_results = True
        return GW_results