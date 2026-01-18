from __future__ import annotations
import os
import re
from typing import TYPE_CHECKING
from monty.io import reverse_readline
from monty.itertools import chunks
from monty.json import MSONable
from monty.serialization import zopen
from pymatgen.core.structure import Molecule
def _parse_logfile(self, logfile):
    """Parse the formatted logfile."""
    cycle_patt = re.compile('Coordinates\\sin\\sGeometry\\sCycle\\s(\\d+)')
    coord_patt = re.compile('\\s+([0-9]+)\\.([A-Za-z]+)' + 3 * '\\s+([-\\.0-9]+)')
    energy_patt = re.compile('<.*>\\s<.*>\\s+current\\senergy\\s+([-\\.0-9]+)\\sHartree')
    final_energy_patt = re.compile('<.*>\\s<.*>\\s+Bond\\sEnergy\\s+([-\\.0-9]+)\\sa\\.u\\.')
    error_patt = re.compile('<.*>\\s<.*>\\s+ERROR\\sDETECTED:\\s(.*)')
    run_type_patt = re.compile('<.*>\\s<.*>\\s+RunType\\s+:\\s(.*)')
    end_patt = re.compile('<.*>\\s<.*>\\s+END')
    parse_cycle = False
    sites = []
    last_cycle = -1
    parse_final = False
    with zopen(logfile, mode='rt') as file:
        for line in reverse_readline(file):
            if line == '':
                continue
            if end_patt.search(line) is None:
                self.is_internal_crash = True
                self.error = 'Internal crash. TAPE13 is generated!'
                self.is_failed = True
                return
            break
    with open(logfile) as file:
        for line in file:
            m = error_patt.search(line)
            if m:
                self.is_failed = True
                self.error = m.group(1)
                break
            if self.run_type is None:
                m = run_type_patt.search(line)
                if m:
                    if m.group(1) == 'FREQUENCIES':
                        self.freq_type = 'Numerical'
                        self.run_type = 'NumericalFreq'
                    elif m.group(1) == 'GEOMETRY OPTIMIZATION':
                        self.run_type = 'GeometryOptimization'
                    elif m.group(1) == 'CREATE':
                        self.run_type = None
                    elif m.group(1) == 'SINGLE POINT':
                        self.run_type = 'SinglePoint'
                    else:
                        raise AdfOutputError('Undefined Runtype!')
            elif self.run_type == 'SinglePoint':
                m = coord_patt.search(line)
                if m:
                    sites.append([m.groups()[0], list(map(float, m.groups()[2:]))])
                else:
                    m = final_energy_patt.search(line)
                    if m:
                        self.final_energy = float(m.group(1))
                        self.final_structure = self._sites_to_mol(sites)
            elif self.run_type == 'GeometryOptimization':
                m = cycle_patt.search(line)
                if m:
                    cycle = int(m.group(1))
                    if cycle <= 0:
                        raise AdfOutputError(f'Wrong cycle={cycle!r}')
                    if cycle > last_cycle:
                        parse_cycle = True
                        last_cycle = cycle
                    else:
                        parse_final = True
                elif parse_cycle:
                    m = coord_patt.search(line)
                    if m:
                        sites.append([m.groups()[1], list(map(float, m.groups()[2:]))])
                    else:
                        m = energy_patt.search(line)
                        if m:
                            self.energies.append(float(m.group(1)))
                            mol = self._sites_to_mol(sites)
                            self.structures.append(mol)
                            parse_cycle = False
                            sites = []
                elif parse_final:
                    m = final_energy_patt.search(line)
                    if m:
                        self.final_energy = float(m.group(1))
            elif self.run_type == 'NumericalFreq':
                break
    if not self.is_failed:
        if self.run_type == 'GeometryOptimization':
            if len(self.structures) > 0:
                self.final_structure = self.structures[-1]
            if self.final_energy is None:
                raise AdfOutputError('The final energy can not be read!')
        elif self.run_type == 'SinglePoint':
            if self.final_structure is None:
                raise AdfOutputError('The final structure is missing!')
            if self.final_energy is None:
                raise AdfOutputError('The final energy can not be read!')