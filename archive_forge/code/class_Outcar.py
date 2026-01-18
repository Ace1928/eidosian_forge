from __future__ import annotations
import datetime
import itertools
import logging
import math
import os
import re
import warnings
import xml.etree.ElementTree as ET
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from glob import glob
from io import StringIO
from pathlib import Path
from typing import TYPE_CHECKING, Literal
import numpy as np
from monty.io import reverse_readfile, zopen
from monty.json import MSONable, jsanitize
from monty.os.path import zpath
from monty.re import regrep
from numpy.testing import assert_allclose
from pymatgen.core import Composition, Element, Lattice, Structure
from pymatgen.core.units import unitized
from pymatgen.electronic_structure.bandstructure import (
from pymatgen.electronic_structure.core import Magmom, Orbital, OrbitalType, Spin
from pymatgen.electronic_structure.dos import CompleteDos, Dos
from pymatgen.entries.computed_entries import ComputedEntry, ComputedStructureEntry
from pymatgen.io.common import VolumetricData as BaseVolumetricData
from pymatgen.io.core import ParseError
from pymatgen.io.vasp.inputs import Incar, Kpoints, Poscar, Potcar
from pymatgen.io.wannier90 import Unk
from pymatgen.util.io_utils import clean_lines, micro_pyawk
from pymatgen.util.num import make_symmetric_matrix_from_upper_tri
class Outcar:
    """
    Parser for data in OUTCAR that is not available in Vasprun.xml.

    Note, this class works a bit differently than most of the other
    VaspObjects, since the OUTCAR can be very different depending on which
    "type of run" performed.

    Creating the OUTCAR class with a filename reads "regular parameters" that
    are always present.

    Attributes:
        magnetization (tuple): Magnetization on each ion as a tuple of dict, e.g.,
            ({"d": 0.0, "p": 0.003, "s": 0.002, "tot": 0.005}, ... )
        chemical_shielding (dict): Chemical shielding on each ion as a dictionary with core and valence contributions.
        unsym_cs_tensor (list): Unsymmetrized chemical shielding tensor matrixes on each ion as a list.
            e.g., [[[sigma11, sigma12, sigma13], [sigma21, sigma22, sigma23], [sigma31, sigma32, sigma33]], ...]
        cs_g0_contribution (np.array): G=0 contribution to chemical shielding. 2D rank 3 matrix.
        cs_core_contribution (dict): Core contribution to chemical shielding. dict. e.g.,
            {'Mg': -412.8, 'C': -200.5, 'O': -271.1}
        efg (tuple): Electric Field Gradient (EFG) tensor on each ion as a tuple of dict, e.g.,
            ({"cq": 0.1, "eta", 0.2, "nuclear_quadrupole_moment": 0.3}, {"cq": 0.7, "eta", 0.8,
            "nuclear_quadrupole_moment": 0.9}, ...)
        charge (tuple): Charge on each ion as a tuple of dict, e.g.,
            ({"p": 0.154, "s": 0.078, "d": 0.0, "tot": 0.232}, ...)
        is_stopped (bool): True if OUTCAR is from a stopped run (using STOPCAR, see VASP Manual).
        run_stats (dict): Various useful run stats as a dict including "System time (sec)", "Total CPU time used (sec)",
            "Elapsed time (sec)", "Maximum memory used (kb)", "Average memory used (kb)", "User time (sec)", "cores".
        elastic_tensor (np.array): Total elastic moduli (Kbar) is given in a 6x6 array matrix.
        drift (np.array): Total drift for each step in eV/Atom.
        ngf (tuple): Dimensions for the Augmentation grid.
        sampling_radii (np.array): Size of the sampling radii in VASP for the test charges for the electrostatic
            potential at each atom. Total array size is the number of elements present in the calculation.
        electrostatic_potential (np.array): Average electrostatic potential at each atomic position in order of
            the atoms in POSCAR.
        final_energy_contribs (dict): Individual contributions to the total final energy as a dictionary.
            Include contributions from keys, e.g.:
            {'DENC': -505778.5184347, 'EATOM': 15561.06492564, 'EBANDS': -804.53201231, 'EENTRO': -0.08932659,
            'EXHF': 0.0, 'Ediel_sol': 0.0, 'PAW double counting': 664.6726974100002, 'PSCENC': 742.48691646,
            'TEWEN': 489742.86847338, 'XCENC': -169.64189814}
        efermi (float): Fermi energy.
        filename (str): Filename.
        final_energy (float): Final energy after extrapolation of sigma back to 0, i.e. energy(sigma->0).
        final_energy_wo_entrp (float): Final energy before extrapolation of sigma, i.e. energy without entropy.
        final_fr_energy (float): Final "free energy", i.e. free energy TOTEN.
        has_onsite_density_matrices (bool): Boolean for if onsite density matrices have been set.
        lcalcpol (bool): If LCALCPOL has been set.
        lepsilon (bool): If LEPSILON has been set.
        nelect (float): Returns the number of electrons in the calculation.
        spin (bool): If spin-polarization was enabled via ISPIN.
        total_mag (float): Total magnetization (in terms of the number of unpaired electrons).

    One can then call a specific reader depending on the type of run being
    performed. These are currently: read_igpar(), read_lepsilon() and
    read_lcalcpol(), read_core_state_eign(), read_avg_core_pot().

    See the documentation of those methods for more documentation.

    Authors: Rickard Armiento, Shyue Ping Ong
    """

    def __init__(self, filename):
        """
        Args:
            filename (str): OUTCAR filename to parse.
        """
        self.filename = filename
        self.is_stopped = False
        serial_compilation = False
        charge = []
        mag_x = []
        mag_y = []
        mag_z = []
        header = []
        run_stats = {}
        total_mag = nelect = efermi = e_fr_energy = e_wo_entrp = e0 = None
        time_patt = re.compile('\\((sec|kb)\\)')
        efermi_patt = re.compile('E-fermi\\s*:\\s*(\\S+)')
        nelect_patt = re.compile('number of electron\\s+(\\S+)\\s+magnetization')
        mag_patt = re.compile('number of electron\\s+\\S+\\s+magnetization\\s+(\\S+)')
        e_fr_energy_pattern = re.compile('free  energy   TOTEN\\s+=\\s+([\\d\\-\\.]+)')
        e_wo_entrp_pattern = re.compile('energy  without entropy\\s*=\\s+([\\d\\-\\.]+)')
        e0_pattern = re.compile('energy\\(sigma->0\\)\\s*=\\s+([\\d\\-\\.]+)')
        all_lines = []
        for line in reverse_readfile(self.filename):
            clean = line.strip()
            all_lines.append(clean)
            if clean.find('soft stop encountered!  aborting job') != -1:
                self.is_stopped = True
            else:
                if time_patt.search(line):
                    tok = line.strip().split(':')
                    try:
                        run_stats[tok[0].strip()] = float(tok[1].strip())
                    except ValueError:
                        run_stats[tok[0].strip()] = None
                    continue
                m = efermi_patt.search(clean)
                if m:
                    try:
                        efermi = float(m.group(1))
                        continue
                    except ValueError:
                        efermi = None
                        continue
                m = nelect_patt.search(clean)
                if m:
                    nelect = float(m.group(1))
                m = mag_patt.search(clean)
                if m:
                    total_mag = float(m.group(1))
                if e_fr_energy is None:
                    m = e_fr_energy_pattern.search(clean)
                    if m:
                        e_fr_energy = float(m.group(1))
                if e_wo_entrp is None:
                    m = e_wo_entrp_pattern.search(clean)
                    if m:
                        e_wo_entrp = float(m.group(1))
                if e0 is None:
                    m = e0_pattern.search(clean)
                    if m:
                        e0 = float(m.group(1))
            if all([nelect, total_mag is not None, efermi is not None, run_stats]):
                break
        read_charge = False
        read_mag_x = False
        read_mag_y = False
        read_mag_z = False
        all_lines.reverse()
        for clean in all_lines:
            if read_charge or read_mag_x or read_mag_y or read_mag_z:
                if clean.startswith('# of ion'):
                    header = re.split('\\s{2,}', clean.strip())
                    header.pop(0)
                else:
                    m = re.match('\\s*(\\d+)\\s+(([\\d\\.\\-]+)\\s+)+', clean)
                    if m:
                        tokens = [float(i) for i in re.findall('[\\d\\.\\-]+', clean)]
                        tokens.pop(0)
                        if read_charge:
                            charge.append(dict(zip(header, tokens)))
                        elif read_mag_x:
                            mag_x.append(dict(zip(header, tokens)))
                        elif read_mag_y:
                            mag_y.append(dict(zip(header, tokens)))
                        elif read_mag_z:
                            mag_z.append(dict(zip(header, tokens)))
                    elif clean.startswith('tot'):
                        read_charge = False
                        read_mag_x = False
                        read_mag_y = False
                        read_mag_z = False
            if clean == 'total charge':
                charge = []
                read_charge = True
                read_mag_x, read_mag_y, read_mag_z = (False, False, False)
            elif clean == 'magnetization (x)':
                mag_x = []
                read_mag_x = True
                read_charge, read_mag_y, read_mag_z = (False, False, False)
            elif clean == 'magnetization (y)':
                mag_y = []
                read_mag_y = True
                read_charge, read_mag_x, read_mag_z = (False, False, False)
            elif clean == 'magnetization (z)':
                mag_z = []
                read_mag_z = True
                read_charge, read_mag_x, read_mag_y = (False, False, False)
            elif re.search('electrostatic', clean):
                read_charge, read_mag_x, read_mag_y, read_mag_z = (False, False, False, False)
        if mag_y and mag_z:
            mag = []
            for idx in range(len(mag_x)):
                mag.append({key: Magmom([mag_x[idx][key], mag_y[idx][key], mag_z[idx][key]]) for key in mag_x[0]})
        else:
            mag = mag_x
        run_stats['cores'] = None
        with zopen(filename, mode='rt') as file:
            for line in file:
                if 'serial' in line:
                    run_stats['cores'] = 1
                    serial_compilation = True
                    break
                if 'running' in line:
                    if line.split()[1] == 'on':
                        run_stats['cores'] = int(line.split()[2])
                    else:
                        run_stats['cores'] = int(line.split()[1])
                    break
        self.run_stats = run_stats
        self.magnetization = tuple(mag)
        self.charge = tuple(charge)
        self.efermi = efermi
        self.nelect = nelect
        self.total_mag = total_mag
        self.final_energy = e0
        self.final_energy_wo_entrp = e_wo_entrp
        self.final_fr_energy = e_fr_energy
        self.data = {}
        self.read_pattern({'nplwv': 'total plane-waves  NPLWV =\\s+(\\*{6}|\\d+)'}, terminate_on_match=True)
        try:
            self.data['nplwv'] = [[int(self.data['nplwv'][0][0])]]
        except ValueError:
            self.data['nplwv'] = [[None]]
        nplwvs_at_kpoints = [n for [n] in self.read_table_pattern('\\n{3}-{104}\\n{3}', '.+plane waves:\\s+(\\*{6,}|\\d+)', 'maximum number of plane-waves' if serial_compilation else 'maximum and minimum number of plane-waves', last_one_only=False, first_one_only=True)]
        self.data['nplwvs_at_kpoints'] = [None for n in nplwvs_at_kpoints]
        for n, nplwv in enumerate(nplwvs_at_kpoints):
            try:
                self.data['nplwvs_at_kpoints'][n] = int(nplwv)
            except ValueError:
                pass
        self.read_pattern({'drift': 'total drift:\\s+([\\.\\-\\d]+)\\s+([\\.\\-\\d]+)\\s+([\\.\\-\\d]+)'}, terminate_on_match=False, postprocess=float)
        self.drift = self.data.get('drift', [])
        self.spin = False
        self.read_pattern({'spin': 'ISPIN  =      2'})
        if self.data.get('spin', []):
            self.spin = True
        self.noncollinear = False
        self.read_pattern({'noncollinear': 'LNONCOLLINEAR =      T'})
        if self.data.get('noncollinear', []):
            self.noncollinear = False
        self.dfpt = False
        self.read_pattern({'ibrion': 'IBRION =\\s+([\\-\\d]+)'}, terminate_on_match=True, postprocess=int)
        if self.data.get('ibrion', [[0]])[0][0] > 6:
            self.dfpt = True
            self.read_internal_strain_tensor()
        self.lepsilon = False
        self.read_pattern({'epsilon': 'LEPSILON=     T'})
        if self.data.get('epsilon', []):
            self.lepsilon = True
            self.read_lepsilon()
            if self.dfpt:
                self.read_lepsilon_ionic()
        self.lcalcpol = False
        self.read_pattern({'calcpol': 'LCALCPOL   =     T'})
        if self.data.get('calcpol', []):
            self.lcalcpol = True
            self.read_lcalcpol()
            self.read_pseudo_zval()
        self.electrostatic_potential = self.ngf = self.sampling_radii = None
        self.read_pattern({'electrostatic': 'average \\(electrostatic\\) potential at core'})
        if self.data.get('electrostatic', []):
            self.read_electrostatic_potential()
        self.nmr_cs = False
        self.read_pattern({'nmr_cs': 'LCHIMAG   =     (T)'})
        if self.data.get('nmr_cs'):
            self.nmr_cs = True
            self.read_chemical_shielding()
            self.read_cs_g0_contribution()
            self.read_cs_core_contribution()
            self.read_cs_raw_symmetrized_tensors()
        self.nmr_efg = False
        self.read_pattern({'nmr_efg': 'NMR quadrupolar parameters'})
        if self.data.get('nmr_efg'):
            self.nmr_efg = True
            self.read_nmr_efg()
            self.read_nmr_efg_tensor()
        self.has_onsite_density_matrices = False
        self.read_pattern({'has_onsite_density_matrices': 'onsite density matrix'}, terminate_on_match=True)
        if 'has_onsite_density_matrices' in self.data:
            self.has_onsite_density_matrices = True
            self.read_onsite_density_matrices()
        final_energy_contribs = {}
        for key in ['PSCENC', 'TEWEN', 'DENC', 'EXHF', 'XCENC', 'PAW double counting', 'EENTRO', 'EBANDS', 'EATOM', 'Ediel_sol']:
            if key == 'PAW double counting':
                self.read_pattern({key: f'{key}\\s+=\\s+([\\.\\-\\d]+)\\s+([\\.\\-\\d]+)'})
            else:
                self.read_pattern({key: f'{key}\\s+=\\s+([\\d\\-\\.]+)'})
            if not self.data[key]:
                continue
            final_energy_contribs[key] = sum(map(float, self.data[key][-1]))
        self.final_energy_contribs = final_energy_contribs

    def read_pattern(self, patterns, reverse=False, terminate_on_match=False, postprocess=str):
        """
        General pattern reading. Uses monty's regrep method. Takes the same
        arguments.

        Args:
            patterns (dict): A dict of patterns, e.g.,
                {"energy": r"energy\\\\(sigma->0\\\\)\\\\s+=\\\\s+([\\\\d\\\\-.]+)"}.
            reverse (bool): Read files in reverse. Defaults to false. Useful for
                large files, esp OUTCARs, especially when used with
                terminate_on_match.
            terminate_on_match (bool): Whether to terminate when there is at
                least one match in each key in pattern.
            postprocess (callable): A post processing function to convert all
                matches. Defaults to str, i.e., no change.

        Renders accessible:
            Any attribute in patterns. For example,
            {"energy": r"energy\\\\(sigma->0\\\\)\\\\s+=\\\\s+([\\\\d\\\\-.]+)"} will set the
            value of self.data["energy"] = [[-1234], [-3453], ...], to the
            results from regex and postprocess. Note that the returned values
            are lists of lists, because you can grep multiple items on one line.
        """
        matches = regrep(self.filename, patterns, reverse=reverse, terminate_on_match=terminate_on_match, postprocess=postprocess)
        for k in patterns:
            self.data[k] = [i[0] for i in matches.get(k, [])]

    def read_table_pattern(self, header_pattern, row_pattern, footer_pattern, postprocess=str, attribute_name=None, last_one_only=True, first_one_only=False):
        """
        Parse table-like data. A table composes of three parts: header,
        main body, footer. All the data matches "row pattern" in the main body
        will be returned.

        Args:
            header_pattern (str): The regular expression pattern matches the
                table header. This pattern should match all the text
                immediately before the main body of the table. For multiple
                sections table match the text until the section of
                interest. MULTILINE and DOTALL options are enforced, as a
                result, the "." meta-character will also match "\\n" in this
                section.
            row_pattern (str): The regular expression matches a single line in
                the table. Capture interested field using regular expression
                groups.
            footer_pattern (str): The regular expression matches the end of the
                table. E.g. a long dash line.
            postprocess (callable): A post processing function to convert all
                matches. Defaults to str, i.e., no change.
            attribute_name (str): Name of this table. If present the parsed data
                will be attached to "data. e.g. self.data["efg"] = [...]
            last_one_only (bool): All the tables will be parsed, if this option
                is set to True, only the last table will be returned. The
                enclosing list will be removed. i.e. Only a single table will
                be returned. Default to be True. Incompatible with first_one_only.
            first_one_only (bool): Only the first occurrence of the table will be
                parsed and the parsing procedure will stop. The enclosing list
                will be removed. i.e. Only a single table will be returned.
                Incompatible with last_one_only.

        Returns:
            List of tables. 1) A table is a list of rows. 2) A row if either a list of
            attribute values in case the capturing group is defined without name in
            row_pattern, or a dict in case that named capturing groups are defined by
            row_pattern.
        """
        if last_one_only and first_one_only:
            raise ValueError('last_one_only and first_one_only options are incompatible')
        with zopen(self.filename, mode='rt') as file:
            text = file.read()
        table_pattern_text = header_pattern + '\\s*^(?P<table_body>(?:\\s+' + row_pattern + ')+)\\s+' + footer_pattern
        table_pattern = re.compile(table_pattern_text, re.MULTILINE | re.DOTALL)
        rp = re.compile(row_pattern)
        tables = []
        for mt in table_pattern.finditer(text):
            table_body_text = mt.group('table_body')
            table_contents = []
            for line in table_body_text.split('\n'):
                ml = rp.search(line)
                if not ml:
                    continue
                d = ml.groupdict()
                if len(d) > 0:
                    processed_line = {k: postprocess(v) for k, v in d.items()}
                else:
                    processed_line = [postprocess(v) for v in ml.groups()]
                table_contents.append(processed_line)
            tables.append(table_contents)
            if first_one_only:
                break
        retained_data = tables[-1] if last_one_only or first_one_only else tables
        if attribute_name is not None:
            self.data[attribute_name] = retained_data
        return retained_data

    def read_electrostatic_potential(self):
        """Parses the eletrostatic potential for the last ionic step."""
        pattern = {'ngf': '\\s+dimension x,y,z NGXF=\\s+([\\.\\-\\d]+)\\sNGYF=\\s+([\\.\\-\\d]+)\\sNGZF=\\s+([\\.\\-\\d]+)'}
        self.read_pattern(pattern, postprocess=int)
        self.ngf = self.data.get('ngf', [[]])[0]
        pattern = {'radii': 'the test charge radii are((?:\\s+[\\.\\-\\d]+)+)'}
        self.read_pattern(pattern, reverse=True, terminate_on_match=True, postprocess=str)
        self.sampling_radii = [*map(float, self.data['radii'][0][0].split())]
        header_pattern = '\\(the norm of the test charge is\\s+[\\.\\-\\d]+\\)'
        table_pattern = '((?:\\s+\\d+\\s*[\\.\\-\\d]+)+)'
        footer_pattern = '\\s+E-fermi :'
        pots = self.read_table_pattern(header_pattern, table_pattern, footer_pattern)
        pots = ''.join(itertools.chain.from_iterable(pots))
        pots = re.findall('\\s+\\d+\\s*([\\.\\-\\d]+)+', pots)
        self.electrostatic_potential = [*map(float, pots)]

    @staticmethod
    def _parse_sci_notation(line):
        """
        Method to parse lines with values in scientific notation and potentially
        without spaces in between the values. This assumes that the scientific
        notation always lists two digits for the exponent, e.g. 3.535E-02

        Args:
            line: line to parse.

        Returns:
            list[float]: numbers if found, empty ist if not
        """
        m = re.findall('[\\.\\-\\d]+E[\\+\\-]\\d{2}', line)
        if m:
            return [float(t) for t in m]
        return []

    def read_freq_dielectric(self):
        """
        Parses the frequency dependent dielectric function (obtained with
        LOPTICS). Frequencies (in eV) are in self.frequencies, and dielectric
        tensor function is given as self.dielectric_tensor_function.
        """
        plasma_pattern = 'plasma frequency squared.*'
        dielectric_pattern = 'frequency dependent\\s+IMAGINARY DIELECTRIC FUNCTION \\(independent particle, no local field effects\\)(\\sdensity-density)*$'
        row_pattern = '\\s+'.join(['([\\.\\-\\d]+)'] * 3)
        plasma_frequencies = defaultdict(list)
        read_plasma = False
        read_dielectric = False
        energies = []
        data = {'REAL': [], 'IMAGINARY': []}
        count = 0
        component = 'IMAGINARY'
        with zopen(self.filename, mode='rt') as file:
            for line in file:
                line = line.strip()
                if re.match(plasma_pattern, line):
                    read_plasma = 'intraband' if 'intraband' in line else 'interband'
                elif re.match(dielectric_pattern, line):
                    read_plasma = False
                    read_dielectric = True
                    row_pattern = '\\s+'.join(['([\\.\\-\\d]+)'] * 7)
                if read_plasma and re.match(row_pattern, line):
                    plasma_frequencies[read_plasma].append([float(t) for t in line.strip().split()])
                elif read_plasma and Outcar._parse_sci_notation(line):
                    plasma_frequencies[read_plasma].append(Outcar._parse_sci_notation(line))
                elif read_dielectric:
                    tokens = None
                    if re.match(row_pattern, line.strip()):
                        tokens = line.strip().split()
                    elif Outcar._parse_sci_notation(line.strip()):
                        tokens = Outcar._parse_sci_notation(line.strip())
                    elif re.match('\\s*-+\\s*', line):
                        count += 1
                    if tokens:
                        if component == 'IMAGINARY':
                            energies.append(float(tokens[0]))
                        xx, yy, zz, xy, yz, xz = (float(t) for t in tokens[1:])
                        matrix = [[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]]
                        data[component].append(matrix)
                    if count == 2:
                        component = 'REAL'
                    elif count == 3:
                        break
        self.plasma_frequencies = {k: np.array(v[:3]) for k, v in plasma_frequencies.items()}
        self.dielectric_energies = np.array(energies)
        self.dielectric_tensor_function = np.array(data['REAL']) + 1j * np.array(data['IMAGINARY'])

    def read_chemical_shielding(self):
        """
        Parse the NMR chemical shieldings data. Only the second part "absolute, valence and core"
        will be parsed. And only the three right most field (ISO_SHIELDING, SPAN, SKEW) will be retrieved.

        Returns:
            List of chemical shieldings in the order of atoms from the OUTCAR. Maryland notation is adopted.
        """
        header_pattern = '\\s+CSA tensor \\(J\\. Mason, Solid State Nucl\\. Magn\\. Reson\\. 2, 285 \\(1993\\)\\)\\s+\\s+-{50,}\\s+\\s+EXCLUDING G=0 CONTRIBUTION\\s+INCLUDING G=0 CONTRIBUTION\\s+\\s+-{20,}\\s+-{20,}\\s+\\s+ATOM\\s+ISO_SHIFT\\s+SPAN\\s+SKEW\\s+ISO_SHIFT\\s+SPAN\\s+SKEW\\s+-{50,}\\s*$'
        first_part_pattern = '\\s+\\(absolute, valence only\\)\\s+$'
        swallon_valence_body_pattern = '.+?\\(absolute, valence and core\\)\\s+$'
        row_pattern = '\\d+(?:\\s+[-]?\\d+\\.\\d+){3}\\s+' + '\\s+'.join(['([-]?\\d+\\.\\d+)'] * 3)
        footer_pattern = '-{50,}\\s*$'
        h1 = header_pattern + first_part_pattern
        cs_valence_only = self.read_table_pattern(h1, row_pattern, footer_pattern, postprocess=float, last_one_only=True)
        h2 = header_pattern + swallon_valence_body_pattern
        cs_valence_and_core = self.read_table_pattern(h2, row_pattern, footer_pattern, postprocess=float, last_one_only=True)
        self.data['chemical_shielding'] = {'valence_only': cs_valence_only, 'valence_and_core': cs_valence_and_core}

    def read_cs_g0_contribution(self):
        """
        Parse the  G0 contribution of NMR chemical shielding.

        Returns:
            G0 contribution matrix as list of list.
        """
        header_pattern = '^\\s+G\\=0 CONTRIBUTION TO CHEMICAL SHIFT \\(field along BDIR\\)\\s+$\\n^\\s+-{50,}$\\n^\\s+BDIR\\s+X\\s+Y\\s+Z\\s*$\\n^\\s+-{50,}\\s*$\\n'
        row_pattern = '(?:\\d+)\\s+' + '\\s+'.join(['([-]?\\d+\\.\\d+)'] * 3)
        footer_pattern = '\\s+-{50,}\\s*$'
        self.read_table_pattern(header_pattern, row_pattern, footer_pattern, postprocess=float, last_one_only=True, attribute_name='cs_g0_contribution')

    def read_cs_core_contribution(self):
        """
        Parse the core contribution of NMR chemical shielding.

        Returns:
            list[list]: G0 contribution matrix.
        """
        header_pattern = '^\\s+Core NMR properties\\s*$\\n\\n^\\s+typ\\s+El\\s+Core shift \\(ppm\\)\\s*$\\n^\\s+-{20,}$\\n'
        row_pattern = '\\d+\\s+(?P<element>[A-Z][a-z]?\\w?)\\s+(?P<shift>[-]?\\d+\\.\\d+)'
        footer_pattern = '\\s+-{20,}\\s*$'
        self.read_table_pattern(header_pattern, row_pattern, footer_pattern, postprocess=str, last_one_only=True, attribute_name='cs_core_contribution')
        core_contrib = {d['element']: float(d['shift']) for d in self.data['cs_core_contribution']}
        self.data['cs_core_contribution'] = core_contrib

    def read_cs_raw_symmetrized_tensors(self):
        """
        Parse the matrix form of NMR tensor before corrected to table.

        Returns:
            nsymmetrized tensors list in the order of atoms.
        """
        header_pattern = '\\s+-{50,}\\s+\\s+Absolute Chemical Shift tensors\\s+\\s+-{50,}$'
        first_part_pattern = '\\s+UNSYMMETRIZED TENSORS\\s+$'
        row_pattern = '\\s+'.join(['([-]?\\d+\\.\\d+)'] * 3)
        unsym_footer_pattern = '^\\s+SYMMETRIZED TENSORS\\s+$'
        with zopen(self.filename, mode='rt') as file:
            text = file.read()
        unsym_table_pattern_text = header_pattern + first_part_pattern + '(?P<table_body>.+)' + unsym_footer_pattern
        table_pattern = re.compile(unsym_table_pattern_text, re.MULTILINE | re.DOTALL)
        rp = re.compile(row_pattern)
        m = table_pattern.search(text)
        if m:
            table_text = m.group('table_body')
            micro_header_pattern = 'ion\\s+\\d+'
            micro_table_pattern_text = micro_header_pattern + '\\s*^(?P<table_body>(?:\\s*' + row_pattern + ')+)\\s+'
            micro_table_pattern = re.compile(micro_table_pattern_text, re.MULTILINE | re.DOTALL)
            unsym_tensors = []
            for mt in micro_table_pattern.finditer(table_text):
                table_body_text = mt.group('table_body')
                tensor_matrix = []
                for line in table_body_text.rstrip().split('\n'):
                    ml = rp.search(line)
                    processed_line = [float(v) for v in ml.groups()]
                    tensor_matrix.append(processed_line)
                unsym_tensors.append(tensor_matrix)
            self.data['unsym_cs_tensor'] = unsym_tensors
        else:
            raise ValueError('NMR UNSYMMETRIZED TENSORS is not found')

    def read_nmr_efg_tensor(self):
        """
        Parses the NMR Electric Field Gradient Raw Tensors.

        Returns:
            A list of Electric Field Gradient Tensors in the order of Atoms from OUTCAR
        """
        header_pattern = 'Electric field gradients \\(V/A\\^2\\)\\n-*\\n ion\\s+V_xx\\s+V_yy\\s+V_zz\\s+V_xy\\s+V_xz\\s+V_yz\\n-*\\n'
        row_pattern = '\\d+\\s+([-\\d\\.]+)\\s+([-\\d\\.]+)\\s+([-\\d\\.]+)\\s+([-\\d\\.]+)\\s+([-\\d\\.]+)\\s+([-\\d\\.]+)'
        footer_pattern = '-*\\n'
        data = self.read_table_pattern(header_pattern, row_pattern, footer_pattern, postprocess=float)
        tensors = [make_symmetric_matrix_from_upper_tri(d) for d in data]
        self.data['unsym_efg_tensor'] = tensors
        return tensors

    def read_nmr_efg(self):
        """
        Parse the NMR Electric Field Gradient interpreted values.

        Returns:
            Electric Field Gradient tensors as a list of dict in the order of atoms from OUTCAR.
            Each dict key/value pair corresponds to a component of the tensors.
        """
        header_pattern = '^\\s+NMR quadrupolar parameters\\s+$\\n^\\s+Cq : quadrupolar parameter\\s+Cq=e[*]Q[*]V_zz/h$\\n^\\s+eta: asymmetry parameters\\s+\\(V_yy - V_xx\\)/ V_zz$\\n^\\s+Q  : nuclear electric quadrupole moment in mb \\(millibarn\\)$\\n^-{50,}$\\n^\\s+ion\\s+Cq\\(MHz\\)\\s+eta\\s+Q \\(mb\\)\\s+$\\n^-{50,}\\s*$\\n'
        row_pattern = '\\d+\\s+(?P<cq>[-]?\\d+\\.\\d+)\\s+(?P<eta>[-]?\\d+\\.\\d+)\\s+(?P<nuclear_quadrupole_moment>[-]?\\d+\\.\\d+)'
        footer_pattern = '-{50,}\\s*$'
        self.read_table_pattern(header_pattern, row_pattern, footer_pattern, postprocess=float, last_one_only=True, attribute_name='efg')

    def read_elastic_tensor(self):
        """
        Parse the elastic tensor data.

        Returns:
            6x6 array corresponding to the elastic tensor from the OUTCAR.
        """
        header_pattern = 'TOTAL ELASTIC MODULI \\(kBar\\)\\s+Direction\\s+([X-Z][X-Z]\\s+)+\\-+'
        row_pattern = '[X-Z][X-Z]\\s+' + '\\s+'.join(['(\\-*[\\.\\d]+)'] * 6)
        footer_pattern = '\\-+'
        et_table = self.read_table_pattern(header_pattern, row_pattern, footer_pattern, postprocess=float)
        self.data['elastic_tensor'] = et_table

    def read_piezo_tensor(self):
        """Parse the piezo tensor data."""
        header_pattern = 'PIEZOELECTRIC TENSOR  for field in x, y, z\\s+\\(C/m\\^2\\)\\s+([X-Z][X-Z]\\s+)+\\-+'
        row_pattern = '[x-z]\\s+' + '\\s+'.join(['(\\-*[\\.\\d]+)'] * 6)
        footer_pattern = 'BORN EFFECTIVE'
        pt_table = self.read_table_pattern(header_pattern, row_pattern, footer_pattern, postprocess=float)
        self.data['piezo_tensor'] = pt_table

    def read_onsite_density_matrices(self):
        """
        Parse the onsite density matrices, returns list with index corresponding
        to atom index in Structure.
        """
        header_pattern = 'spin component  1\\n'
        row_pattern = '[^\\S\\r\\n]*(?:(-?[\\d.]+))' + '(?:[^\\S\\r\\n]*(-?[\\d.]+)[^\\S\\r\\n]*)?' * 6 + '.*?'
        footer_pattern = '\\nspin component  2'
        spin1_component = self.read_table_pattern(header_pattern, row_pattern, footer_pattern, postprocess=lambda x: float(x) if x else None, last_one_only=False)
        spin1_component = [[[e for e in row if e is not None] for row in matrix] for matrix in spin1_component]
        header_pattern = 'spin component  2\\n'
        row_pattern = '[^\\S\\r\\n]*(?:([\\d.-]+))' + '(?:[^\\S\\r\\n]*(-?[\\d.]+)[^\\S\\r\\n]*)?' * 6 + '.*?'
        footer_pattern = '\\n occupancies and eigenvectors'
        spin2_component = self.read_table_pattern(header_pattern, row_pattern, footer_pattern, postprocess=lambda x: float(x) if x else None, last_one_only=False)
        spin2_component = [[[e for e in row if e is not None] for row in matrix] for matrix in spin2_component]
        self.data['onsite_density_matrices'] = [{Spin.up: spin1_component[idx], Spin.down: spin2_component[idx]} for idx in range(len(spin1_component))]

    def read_corrections(self, reverse=True, terminate_on_match=True):
        """
        Reads the dipol qudropol corrections into the
        Outcar.data["dipol_quadrupol_correction"].

        Args:
            reverse (bool): Whether to start from end of OUTCAR. Defaults to True.
            terminate_on_match (bool): Whether to terminate once match is found. Defaults to True.
        """
        patterns = {'dipol_quadrupol_correction': 'dipol\\+quadrupol energy correction\\s+([\\d\\-\\.]+)'}
        self.read_pattern(patterns, reverse=reverse, terminate_on_match=terminate_on_match, postprocess=float)
        self.data['dipol_quadrupol_correction'] = self.data['dipol_quadrupol_correction'][0][0]

    def read_neb(self, reverse=True, terminate_on_match=True):
        """
        Reads NEB data. This only works with OUTCARs from both normal
        VASP NEB calculations or from the CI NEB method implemented by
        Henkelman et al.

        Args:
            reverse (bool): Read files in reverse. Defaults to false. Useful for
                large files, esp OUTCARs, especially when used with
                terminate_on_match. Defaults to True here since we usually
                want only the final value.
            terminate_on_match (bool): Whether to terminate when there is at
                least one match in each key in pattern. Defaults to True here
                since we usually want only the final value.

        Renders accessible:
            tangent_force - Final tangent force.
            energy - Final energy.
            These can be accessed under Outcar.data[key]
        """
        patterns = {'energy': 'energy\\(sigma->0\\)\\s+=\\s+([\\d\\-\\.]+)', 'tangent_force': '(NEB: projections on to tangent \\(spring, REAL\\)\\s+\\S+|tangential force \\(eV/A\\))\\s+([\\d\\-\\.]+)'}
        self.read_pattern(patterns, reverse=reverse, terminate_on_match=terminate_on_match, postprocess=str)
        self.data['energy'] = float(self.data['energy'][0][0])
        if self.data.get('tangent_force'):
            self.data['tangent_force'] = float(self.data['tangent_force'][0][1])

    def read_igpar(self):
        """
        Renders accessible:
            er_ev = e<r>_ev (dictionary with Spin.up/Spin.down as keys)
            er_bp = e<r>_bp (dictionary with Spin.up/Spin.down as keys)
            er_ev_tot = spin up + spin down summed
            er_bp_tot = spin up + spin down summed
            p_elc = spin up + spin down summed
            p_ion = spin up + spin down summed.

        (See VASP section "LBERRY,  IGPAR,  NPPSTR,  DIPOL tags" for info on
        what these are).
        """
        self.er_ev = {}
        self.er_bp = {}
        self.er_ev_tot = None
        self.er_bp_tot = None
        self.p_elec = self.p_ion = None
        try:
            search = []

            def er_ev(results, match):
                results.er_ev[Spin.up] = np.array(map(float, match.groups()[1:4])) / 2
                results.er_ev[Spin.down] = results.er_ev[Spin.up]
                results.context = 2
            search.append(['^ *e<r>_ev=\\( *([-0-9.Ee+]*) *([-0-9.Ee+]*) *([-0-9.Ee+]*) *\\)', None, er_ev])

            def er_bp(results, match):
                results.er_bp[Spin.up] = np.array([float(match.group(i)) for i in range(1, 4)]) / 2
                results.er_bp[Spin.down] = results.er_bp[Spin.up]
            search.append(['^ *e<r>_bp=\\( *([-0-9.Ee+]*) *([-0-9.Ee+]*) *([-0-9.Ee+]*) *\\)', lambda results, _line: results.context == 2, er_bp])

            def er_ev_up(results, match):
                results.er_ev[Spin.up] = np.array([float(match.group(i)) for i in range(1, 4)])
                results.context = Spin.up
            search.append(['^.*Spin component 1 *e<r>_ev=\\( *([-0-9.Ee+]*) *([-0-9.Ee+]*) *([-0-9.Ee+]*) *\\)', None, er_ev_up])

            def er_bp_up(results, match):
                results.er_bp[Spin.up] = np.array([float(match.group(1)), float(match.group(2)), float(match.group(3))])
            search.append(['^ *e<r>_bp=\\( *([-0-9.Ee+]*) *([-0-9.Ee+]*) *([-0-9.Ee+]*) *\\)', lambda results, _line: results.context == Spin.up, er_bp_up])

            def er_ev_dn(results, match):
                results.er_ev[Spin.down] = np.array([float(match.group(1)), float(match.group(2)), float(match.group(3))])
                results.context = Spin.down
            search.append(['^.*Spin component 2 *e<r>_ev=\\( *([-0-9.Ee+]*) *([-0-9.Ee+]*) *([-0-9.Ee+]*) *\\)', None, er_ev_dn])

            def er_bp_dn(results, match):
                results.er_bp[Spin.down] = np.array([float(match.group(i)) for i in range(1, 4)])
            search.append(['^ *e<r>_bp=\\( *([-0-9.Ee+]*) *([-0-9.Ee+]*) *([-0-9.Ee+]*) *\\)', lambda results, _line: results.context == Spin.down, er_bp_dn])

            def p_elc(results, match):
                results.p_elc = np.array([float(match.group(i)) for i in range(1, 4)])
            search.append(['^.*Total electronic dipole moment: *p\\[elc\\]=\\( *([-0-9.Ee+]*) *([-0-9.Ee+]*) *([-0-9.Ee+]*) *\\)', None, p_elc])

            def p_ion(results, match):
                results.p_ion = np.array([float(match.group(i)) for i in range(1, 4)])
            search.append(['^.*ionic dipole moment: *p\\[ion\\]=\\( *([-0-9.Ee+]*) *([-0-9.Ee+]*) *([-0-9.Ee+]*) *\\)', None, p_ion])
            self.context = None
            self.er_ev = {Spin.up: None, Spin.down: None}
            self.er_bp = {Spin.up: None, Spin.down: None}
            micro_pyawk(self.filename, search, self)
            if self.er_ev[Spin.up] is not None and self.er_ev[Spin.down] is not None:
                self.er_ev_tot = self.er_ev[Spin.up] + self.er_ev[Spin.down]
            if self.er_bp[Spin.up] is not None and self.er_bp[Spin.down] is not None:
                self.er_bp_tot = self.er_bp[Spin.up] + self.er_bp[Spin.down]
        except Exception:
            raise RuntimeError('IGPAR OUTCAR could not be parsed.')

    def read_internal_strain_tensor(self):
        """
        Reads the internal strain tensor and populates self.internal_strain_tensor with an array of voigt notation
            tensors for each site.
        """
        search = []

        def internal_strain_start(results, match):
            results.internal_strain_ion = int(match.group(1)) - 1
            results.internal_strain_tensor.append(np.zeros((3, 6)))
        search.append(['INTERNAL STRAIN TENSOR FOR ION\\s+(\\d+)\\s+for displacements in x,y,z  \\(eV/Angst\\):', None, internal_strain_start])

        def internal_strain_data(results, match):
            if match.group(1).lower() == 'x':
                index = 0
            elif match.group(1).lower() == 'y':
                index = 1
            elif match.group(1).lower() == 'z':
                index = 2
            else:
                raise IndexError(f"Couldn't parse row index from symbol for internal strain tensor: {match.group(1)}")
            results.internal_strain_tensor[results.internal_strain_ion][index] = np.array([float(match.group(i)) for i in range(2, 8)])
            if index == 2:
                results.internal_strain_ion = None
        search.append(['^\\s+([x,y,z])\\s+' + '([-]?\\d+\\.\\d+)\\s+' * 6, lambda results, _line: results.internal_strain_ion is not None, internal_strain_data])
        self.internal_strain_ion = None
        self.internal_strain_tensor = []
        micro_pyawk(self.filename, search, self)

    def read_lepsilon(self):
        """
        Reads an LEPSILON run.

        # TODO: Document the actual variables.
        """
        try:
            search = []

            def dielectric_section_start(results, match):
                results.dielectric_index = -1
            search.append(['MACROSCOPIC STATIC DIELECTRIC TENSOR \\(', None, dielectric_section_start])

            def dielectric_section_start2(results, match):
                results.dielectric_index = 0
            search.append(['-------------------------------------', lambda results, _line: results.dielectric_index == -1, dielectric_section_start2])

            def dielectric_data(results, match):
                results.dielectric_tensor[results.dielectric_index, :] = np.array([float(match.group(i)) for i in range(1, 4)])
                results.dielectric_index += 1
            search.append(['^ *([-0-9.Ee+]+) +([-0-9.Ee+]+) +([-0-9.Ee+]+) *$', lambda results, _line: results.dielectric_index >= 0 if results.dielectric_index is not None else None, dielectric_data])

            def dielectric_section_stop(results, match):
                results.dielectric_index = None
            search.append(['-------------------------------------', lambda results, _line: results.dielectric_index >= 1 if results.dielectric_index is not None else None, dielectric_section_stop])
            self.dielectric_index = None
            self.dielectric_tensor = np.zeros((3, 3))

            def piezo_section_start(results, _match):
                results.piezo_index = 0
            search.append(['PIEZOELECTRIC TENSOR  for field in x, y, z        \\(C/m\\^2\\)', None, piezo_section_start])

            def piezo_data(results, match):
                results.piezo_tensor[results.piezo_index, :] = np.array([float(match.group(i)) for i in range(1, 7)])
                results.piezo_index += 1
            search.append(['^ *[xyz] +([-0-9.Ee+]+) +([-0-9.Ee+]+) +([-0-9.Ee+]+) *([-0-9.Ee+]+) +([-0-9.Ee+]+) +([-0-9.Ee+]+)*$', lambda results, _line: results.piezo_index >= 0 if results.piezo_index is not None else None, piezo_data])

            def piezo_section_stop(results, _match):
                results.piezo_index = None
            search.append(['-------------------------------------', lambda results, _line: results.piezo_index >= 1 if results.piezo_index is not None else None, piezo_section_stop])
            self.piezo_index = None
            self.piezo_tensor = np.zeros((3, 6))

            def born_section_start(results, _match):
                results.born_ion = -1
            search.append(['BORN EFFECTIVE CHARGES ', None, born_section_start])

            def born_ion(results, match):
                results.born_ion = int(match.group(1)) - 1
                results.born.append(np.zeros((3, 3)))
            search.append(['ion +([0-9]+)', lambda results, _line: results.born_ion is not None, born_ion])

            def born_data(results, match):
                results.born[results.born_ion][int(match.group(1)) - 1, :] = np.array([float(match.group(i)) for i in range(2, 5)])
            search.append(['^ *([1-3]+) +([-0-9.Ee+]+) +([-0-9.Ee+]+) +([-0-9.Ee+]+)$', lambda results, _line: results.born_ion >= 0 if results.born_ion is not None else results.born_ion, born_data])

            def born_section_stop(results, _match):
                results.born_ion = None
            search.append(['-------------------------------------', lambda results, _line: results.born_ion >= 1 if results.born_ion is not None else results.born_ion, born_section_stop])
            self.born_ion = None
            self.born = []
            micro_pyawk(self.filename, search, self)
            self.born = np.array(self.born)
            self.dielectric_tensor = self.dielectric_tensor.tolist()
            self.piezo_tensor = self.piezo_tensor.tolist()
        except Exception:
            raise RuntimeError('LEPSILON OUTCAR could not be parsed.')

    def read_lepsilon_ionic(self):
        """
        Reads an LEPSILON run, the ionic component.

        # TODO: Document the actual variables.
        """
        try:
            search = []

            def dielectric_section_start(results, _match):
                results.dielectric_ionic_index = -1
            search.append(['MACROSCOPIC STATIC DIELECTRIC TENSOR IONIC', None, dielectric_section_start])

            def dielectric_section_start2(results, _match):
                results.dielectric_ionic_index = 0
            search.append(['-------------------------------------', lambda results, _line: results.dielectric_ionic_index == -1 if results.dielectric_ionic_index is not None else results.dielectric_ionic_index, dielectric_section_start2])

            def dielectric_data(results, match):
                results.dielectric_ionic_tensor[results.dielectric_ionic_index, :] = np.array([float(match.group(i)) for i in range(1, 4)])
                results.dielectric_ionic_index += 1
            search.append(['^ *([-0-9.Ee+]+) +([-0-9.Ee+]+) +([-0-9.Ee+]+) *$', lambda results, _line: results.dielectric_ionic_index >= 0 if results.dielectric_ionic_index is not None else results.dielectric_ionic_index, dielectric_data])

            def dielectric_section_stop(results, _match):
                results.dielectric_ionic_index = None
            search.append(['-------------------------------------', lambda results, _line: results.dielectric_ionic_index >= 1 if results.dielectric_ionic_index is not None else results.dielectric_ionic_index, dielectric_section_stop])
            self.dielectric_ionic_index = None
            self.dielectric_ionic_tensor = np.zeros((3, 3))

            def piezo_section_start(results, _match):
                results.piezo_ionic_index = 0
            search.append(['PIEZOELECTRIC TENSOR IONIC CONTR  for field in x, y, z        ', None, piezo_section_start])

            def piezo_data(results, match):
                results.piezo_ionic_tensor[results.piezo_ionic_index, :] = np.array([float(match.group(i)) for i in range(1, 7)])
                results.piezo_ionic_index += 1
            search.append(['^ *[xyz] +([-0-9.Ee+]+) +([-0-9.Ee+]+) +([-0-9.Ee+]+) *([-0-9.Ee+]+) +([-0-9.Ee+]+) +([-0-9.Ee+]+)*$', lambda results, _line: results.piezo_ionic_index >= 0 if results.piezo_ionic_index is not None else results.piezo_ionic_index, piezo_data])

            def piezo_section_stop(results, _match):
                results.piezo_ionic_index = None
            search.append(['-------------------------------------', lambda results, _line: results.piezo_ionic_index >= 1 if results.piezo_ionic_index is not None else results.piezo_ionic_index, piezo_section_stop])
            self.piezo_ionic_index = None
            self.piezo_ionic_tensor = np.zeros((3, 6))
            micro_pyawk(self.filename, search, self)
            self.dielectric_ionic_tensor = self.dielectric_ionic_tensor.tolist()
            self.piezo_ionic_tensor = self.piezo_ionic_tensor.tolist()
        except Exception:
            raise RuntimeError('ionic part of LEPSILON OUTCAR could not be parsed.')

    def read_lcalcpol(self):
        """
        Reads the lcalpol.

        # TODO: Document the actual variables.
        """
        self.p_elec = self.p_sp1 = self.p_sp2 = self.p_ion = None
        try:
            search = []

            def p_elec(results, match):
                results.p_elec = np.array([float(match.group(1)), float(match.group(2)), float(match.group(3))])
            search.append(['^.*Total electronic dipole moment: *p\\[elc\\]=\\( *([-0-9.Ee+]*) *([-0-9.Ee+]*) *([-0-9.Ee+]*) *\\)', None, p_elec])
            if self.spin and (not self.noncollinear):

                def p_sp1(results, match):
                    results.p_sp1 = np.array([float(match.group(1)), float(match.group(2)), float(match.group(3))])
                search.append(['^.*p\\[sp1\\]=\\( *([-0-9.Ee+]*) *([-0-9.Ee+]*) *([-0-9.Ee+]*) *\\)', None, p_sp1])

                def p_sp2(results, match):
                    results.p_sp2 = np.array([float(match.group(1)), float(match.group(2)), float(match.group(3))])
                search.append(['^.*p\\[sp2\\]=\\( *([-0-9.Ee+]*) *([-0-9.Ee+]*) *([-0-9.Ee+]*) *\\)', None, p_sp2])

            def p_ion(results, match):
                results.p_ion = np.array([float(match.group(1)), float(match.group(2)), float(match.group(3))])
            search.append(['^.*Ionic dipole moment: *p\\[ion\\]=\\( *([-0-9.Ee+]*) *([-0-9.Ee+]*) *([-0-9.Ee+]*) *\\)', None, p_ion])
            micro_pyawk(self.filename, search, self)
            regex = '^.*Ionic dipole moment: .*'
            search = [[regex, None, lambda x, y: x.append(y.group(0))]]
            results = micro_pyawk(self.filename, search, [])
            if '|e|' in results[0]:
                self.p_elec *= -1
                self.p_ion *= -1
                if self.spin and (not self.noncollinear):
                    self.p_sp1 *= -1
                    self.p_sp2 *= -1
        except Exception as exc:
            print(exc.args)
            raise RuntimeError('LCALCPOL OUTCAR could not be parsed.') from exc

    def read_pseudo_zval(self):
        """Create pseudopotential ZVAL dictionary."""
        try:

            def atom_symbols(results, match):
                element_symbol = match.group(1)
                if not hasattr(results, 'atom_symbols'):
                    results.atom_symbols = []
                results.atom_symbols.append(element_symbol.strip())

            def zvals(results, match):
                zvals = match.group(1)
                results.zvals = map(float, re.findall('-?\\d+\\.\\d*', zvals))
            search = []
            search.extend((['(?<=VRHFIN =)(.*)(?=:)', None, atom_symbols], ['^\\s+ZVAL.*=(.*)', None, zvals]))
            micro_pyawk(self.filename, search, self)
            zval_dict = {}
            for x, y in zip(self.atom_symbols, self.zvals):
                zval_dict.update({x: y})
            self.zval_dict = zval_dict
            del self.atom_symbols
            del self.zvals
        except Exception:
            raise RuntimeError('ZVAL dict could not be parsed.')

    def read_core_state_eigen(self):
        """
        Read the core state eigenenergies at each ionic step.

        Returns:
            A list of dict over the atom such as [{"AO":[core state eig]}].
            The core state eigenenergie list for each AO is over all ionic
            step.

        Example:
            The core state eigenenergie of the 2s AO of the 6th atom of the
            structure at the last ionic step is [5]["2s"][-1]
        """
        with zopen(self.filename, mode='rt') as foutcar:
            line = foutcar.readline()
            while line != '':
                line = foutcar.readline()
                if 'NIONS =' in line:
                    natom = int(line.split('NIONS =')[1])
                    cl = [defaultdict(list) for i in range(natom)]
                if 'the core state eigen' in line:
                    iat = -1
                    while line != '':
                        line = foutcar.readline()
                        if 'E-fermi' in line:
                            break
                        data = line.split()
                        if len(data) % 2 == 1:
                            iat += 1
                            data = data[1:]
                        for i in range(0, len(data), 2):
                            cl[iat][data[i]].append(float(data[i + 1]))
        return cl

    def read_avg_core_poten(self):
        """
        Read the core potential at each ionic step.

        Returns:
            A list for each ionic step containing a list of the average core
            potentials for each atom: [[avg core pot]].

        Example:
            The average core potential of the 2nd atom of the structure at the
            last ionic step is: [-1][1]
        """
        with zopen(self.filename, mode='rt') as foutcar:
            line = foutcar.readline()
            aps = []
            while line != '':
                line = foutcar.readline()
                if 'the norm of the test charge is' in line:
                    ap = []
                    while line != '':
                        line = foutcar.readline()
                        if 'E-fermi' in line:
                            aps.append(ap)
                            break
                        npots = int((len(line) - 1) / 17)
                        for i in range(npots):
                            start = i * 17
                            ap.append(float(line[start + 8:start + 17]))
        return aps

    def as_dict(self):
        """MSONable dict."""
        dct = {'@module': type(self).__module__, '@class': type(self).__name__, 'efermi': self.efermi, 'run_stats': self.run_stats, 'magnetization': self.magnetization, 'charge': self.charge, 'total_magnetization': self.total_mag, 'nelect': self.nelect, 'is_stopped': self.is_stopped, 'drift': self.drift, 'ngf': self.ngf, 'sampling_radii': self.sampling_radii, 'electrostatic_potential': self.electrostatic_potential}
        if self.lepsilon:
            dct.update({'piezo_tensor': self.piezo_tensor, 'dielectric_tensor': self.dielectric_tensor, 'born': self.born})
        if self.dfpt:
            dct['internal_strain_tensor'] = self.internal_strain_tensor
        if self.dfpt and self.lepsilon:
            dct.update(piezo_ionic_tensor=self.piezo_ionic_tensor, dielectric_ionic_tensor=self.dielectric_ionic_tensor)
        if self.lcalcpol:
            dct.update({'p_elec': self.p_elec, 'p_ion': self.p_ion})
            if self.spin and (not self.noncollinear):
                dct.update({'p_sp1': self.p_sp1, 'p_sp2': self.p_sp2})
            dct['zval_dict'] = self.zval_dict
        if self.nmr_cs:
            dct.update(nmr_cs={'valence and core': self.data['chemical_shielding']['valence_and_core'], 'valence_only': self.data['chemical_shielding']['valence_only'], 'g0': self.data['cs_g0_contribution'], 'core': self.data['cs_core_contribution'], 'raw': self.data['unsym_cs_tensor']})
        if self.nmr_efg:
            dct.update(nmr_efg={'raw': self.data['unsym_efg_tensor'], 'parameters': self.data['efg']})
        if self.has_onsite_density_matrices:
            onsite_density_matrices = [{str(k): v for k, v in d.items()} for d in self.data['onsite_density_matrices']]
            dct['onsite_density_matrices'] = onsite_density_matrices
        return dct

    def read_fermi_contact_shift(self):
        """
        Output example:
        Fermi contact (isotropic) hyperfine coupling parameter (MHz)
        -------------------------------------------------------------
        ion      A_pw      A_1PS     A_1AE     A_1c      A_tot
        -------------------------------------------------------------
         1      -0.002    -0.002    -0.051     0.000    -0.052
         2      -0.002    -0.002    -0.051     0.000    -0.052
         3       0.056     0.056     0.321    -0.048     0.321
        -------------------------------------------------------------
        , which corresponds to
        [[-0.002, -0.002, -0.051, 0.0, -0.052],
         [-0.002, -0.002, -0.051, 0.0, -0.052],
         [0.056, 0.056, 0.321, -0.048, 0.321]] from 'fch' data.
        """
        header_pattern1 = '\\s*Fermi contact \\(isotropic\\) hyperfine coupling parameter \\(MHz\\)\\s+\\s*\\-+\\s*ion\\s+A_pw\\s+A_1PS\\s+A_1AE\\s+A_1c\\s+A_tot\\s+\\s*\\-+'
        row_pattern1 = '(?:\\d+)\\s+' + '\\s+'.join(['([-]?\\d+\\.\\d+)'] * 5)
        footer_pattern = '\\-+'
        fch_table = self.read_table_pattern(header_pattern1, row_pattern1, footer_pattern, postprocess=float, last_one_only=True)
        header_pattern2 = '\\s*Dipolar hyperfine coupling parameters \\(MHz\\)\\s+\\s*\\-+\\s*ion\\s+A_xx\\s+A_yy\\s+A_zz\\s+A_xy\\s+A_xz\\s+A_yz\\s+\\s*\\-+'
        row_pattern2 = '(?:\\d+)\\s+' + '\\s+'.join(['([-]?\\d+\\.\\d+)'] * 6)
        dh_table = self.read_table_pattern(header_pattern2, row_pattern2, footer_pattern, postprocess=float, last_one_only=True)
        header_pattern3 = '\\s*Total hyperfine coupling parameters after diagonalization \\(MHz\\)\\s+\\s*\\(convention: \\|A_zz\\| > \\|A_xx\\| > \\|A_yy\\|\\)\\s+\\s*\\-+\\s*ion\\s+A_xx\\s+A_yy\\s+A_zz\\s+asymmetry \\(A_yy - A_xx\\)/ A_zz\\s+\\s*\\-+'
        row_pattern3 = '(?:\\d+)\\s+' + '\\s+'.join(['([-]?\\d+\\.\\d+)'] * 4)
        th_table = self.read_table_pattern(header_pattern3, row_pattern3, footer_pattern, postprocess=float, last_one_only=True)
        fc_shift_table = {'fch': fch_table, 'dh': dh_table, 'th': th_table}
        self.data['fermi_contact_shift'] = fc_shift_table