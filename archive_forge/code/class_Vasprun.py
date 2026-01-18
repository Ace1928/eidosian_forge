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
class Vasprun(MSONable):
    """
    Vastly improved cElementTree-based parser for vasprun.xml files. Uses
    iterparse to support incremental parsing of large files.
    Speedup over Dom is at least 2x for smallish files (~1Mb) to orders of
    magnitude for larger files (~10Mb).

    **VASP results**

    Attributes:
        ionic_steps (list): All ionic steps in the run as a list of {"structure": structure at end of run,
            "electronic_steps": {All electronic step data in vasprun file}, "stresses": stress matrix}.
        tdos (Dos): Total dos calculated at the end of run.
        idos (Dos): Integrated dos calculated at the end of run.
        pdos (list): List of list of PDos objects. Access as pdos[atomindex][orbitalindex].
        efermi (float): Fermi energy.
        eigenvalues (dict): Final eigenvalues as a dict of {(spin, kpoint index):[[eigenvalue, occu]]}.
            The kpoint index is 0-based (unlike the 1-based indexing in VASP).
        projected_eigenvalues (dict): Final projected eigenvalues as a dict of {spin: nd-array}.
            To access a particular value, you need to do
            Vasprun.projected_eigenvalues[spin][kpoint index][band index][atom index][orbital_index].
            The kpoint, band and atom indices are 0-based (unlike the 1-based indexing in VASP).
        projected_magnetisation (np.array): Final projected magnetization as a numpy array with the
            shape (nkpoints, nbands, natoms, norbitals, 3). Where the last axis is the contribution in the
            3 Cartesian directions. This attribute is only set if spin-orbit coupling (LSORBIT = True) or
            non-collinear magnetism (LNONCOLLINEAR = True) is turned on in the INCAR.
        other_dielectric (dict): Dictionary, with the tag comment as key, containing other variants of
            the real and imaginary part of the dielectric constant (e.g., computed by RPA) in function of
            the energy (frequency). Optical properties (e.g. absorption coefficient) can be obtained through this.
            The data is given as a tuple of 3 values containing each of them the energy, the real part tensor,
            and the imaginary part tensor ([energies],[[real_partxx,real_partyy,real_partzz,real_partxy,
            real_partyz,real_partxz]],[[imag_partxx,imag_partyy,imag_partzz,imag_partxy, imag_partyz, imag_partxz]]).
        nionic_steps (int): The total number of ionic steps. This number is always equal to the total number
            of steps in the actual run even if ionic_step_skip is used.
        force_constants (np.array): Force constants computed in phonon DFPT run(IBRION = 8).
            The data is a 4D numpy array of shape (natoms, natoms, 3, 3).
        normalmode_eigenvals (np.array): Normal mode frequencies. 1D numpy array of size 3*natoms.
        normalmode_eigenvecs (np.array): Normal mode eigen vectors. 3D numpy array of shape (3*natoms, natoms, 3).
        md_data (list): Available only for ML MD runs, i.e., INCAR with ML_LMLFF = .TRUE. md_data is a list of
            dict with the following format: [{'energy': {'e_0_energy': -525.07195568, 'e_fr_energy': -525.07195568,
            'e_wo_entrp': -525.07195568, 'kinetic': 3.17809233, 'lattice kinetic': 0.0, 'nosekinetic': 1.323e-5,
            'nosepot': 0.0, 'total': -521.89385012}, 'forces': [[0.17677989, 0.48309874, 1.85806696], ...],
            'structure': Structure object}].
        incar (Incar): Incar object for parameters specified in INCAR file.
        parameters (Incar): Incar object with parameters that vasp actually used, including all defaults.
        kpoints (Kpoints): Kpoints object for KPOINTS specified in run.
        actual_kpoints (list): List of actual kpoints, e.g., [[0.25, 0.125, 0.08333333], [-0.25, 0.125, 0.08333333],
            [0.25, 0.375, 0.08333333], ....].
        actual_kpoints_weights (list): List of kpoint weights, E.g., [0.04166667, 0.04166667, 0.04166667, 0.04166667,
            0.04166667, ....].
        atomic_symbols (list): List of atomic symbols, e.g., ["Li", "Fe", "Fe", "P", "P", "P"].
        potcar_symbols (list): List of POTCAR symbols. e.g., ["PAW_PBE Li 17Jan2003", "PAW_PBE Fe 06Sep2000", ..].
        kpoints_opt_props (object): Object whose attributes are the data from KPOINTS_OPT (if present,
            else None). Attributes of the same name have the same format and meaning as Vasprun (or they are
            None if absent). Attributes are: tdos, idos, pdos, efermi, eigenvalues, projected_eigenvalues,
            projected magnetisation, kpoints, actual_kpoints, actual_kpoints_weights, dos_has_errors.

    Author: Shyue Ping Ong
    """

    def __init__(self, filename: str | Path, ionic_step_skip: int | None=None, ionic_step_offset: int=0, parse_dos: bool=True, parse_eigen: bool=True, parse_projected_eigen: bool=False, parse_potcar_file: bool=True, occu_tol: float=1e-08, separate_spins: bool=False, exception_on_bad_xml: bool=True) -> None:
        """
        Args:
            filename (str): Filename to parse
            ionic_step_skip (int): If ionic_step_skip is a number > 1,
                only every ionic_step_skip ionic steps will be read for
                structure and energies. This is very useful if you are parsing
                very large vasprun.xml files and you are not interested in every
                single ionic step. Note that the final energies may not be the
                actual final energy in the vasprun.
            ionic_step_offset (int): Used together with ionic_step_skip. If set,
                the first ionic step read will be offset by the amount of
                ionic_step_offset. For example, if you want to start reading
                every 10th structure but only from the 3rd structure onwards,
                set ionic_step_skip to 10 and ionic_step_offset to 3. Main use
                case is when doing statistical structure analysis with
                extremely long time scale multiple VASP calculations of
                varying numbers of steps.
            parse_dos (bool): Whether to parse the dos. Defaults to True. Set
                to False to shave off significant time from the parsing if you
                are not interested in getting those data.
            parse_eigen (bool): Whether to parse the eigenvalues. Defaults to
                True. Set to False to shave off significant time from the
                parsing if you are not interested in getting those data.
            parse_projected_eigen (bool): Whether to parse the projected
                eigenvalues and magnetization. Defaults to False. Set to True to obtain
                projected eigenvalues and magnetization. **Note that this can take an
                extreme amount of time and memory.** So use this wisely.
            parse_potcar_file (bool/str): Whether to parse the potcar file to read
                the potcar hashes for the potcar_spec attribute. Defaults to True,
                where no hashes will be determined and the potcar_spec dictionaries
                will read {"symbol": ElSymbol, "hash": None}. By Default, looks in
                the same directory as the vasprun.xml, with same extensions as
                Vasprun.xml. If a string is provided, looks at that filepath.
            occu_tol (float): Sets the minimum tol for the determination of the
                vbm and cbm. Usually the default of 1e-8 works well enough,
                but there may be pathological cases.
            separate_spins (bool): Whether the band gap, CBM, and VBM should be
                reported for each individual spin channel. Defaults to False,
                which computes the eigenvalue band properties independent of
                the spin orientation. If True, the calculation must be spin-polarized.
            exception_on_bad_xml (bool): Whether to throw a ParseException if a
                malformed XML is detected. Default to True, which ensures only
                proper vasprun.xml are parsed. You can set to False if you want
                partial results (e.g., if you are monitoring a calculation during a
                run), but use the results with care. A warning is issued.
        """
        self.filename = filename
        self.ionic_step_skip = ionic_step_skip
        self.ionic_step_offset = ionic_step_offset
        self.occu_tol = occu_tol
        self.separate_spins = separate_spins
        self.exception_on_bad_xml = exception_on_bad_xml
        with zopen(filename, mode='rt') as file:
            if ionic_step_skip or ionic_step_offset:
                run = file.read()
                steps = run.split('<calculation>')
                preamble = steps.pop(0)
                self.nionic_steps = len(steps)
                new_steps = steps[ionic_step_offset::int(ionic_step_skip or 1)]
                to_parse = '<calculation>'.join(new_steps)
                if steps[-1] != new_steps[-1]:
                    to_parse = f'{preamble}<calculation>{to_parse}{steps[-1].split('</calculation>')[-1]}'
                else:
                    to_parse = f'{preamble}<calculation>{to_parse}'
                self._parse(StringIO(to_parse), parse_dos=parse_dos, parse_eigen=parse_eigen, parse_projected_eigen=parse_projected_eigen)
            else:
                self._parse(file, parse_dos=parse_dos, parse_eigen=parse_eigen, parse_projected_eigen=parse_projected_eigen)
                self.nionic_steps = len(self.ionic_steps)
            if parse_potcar_file:
                self.update_potcar_spec(parse_potcar_file)
                self.update_charge_from_potcar(parse_potcar_file)
        if self.incar.get('ALGO') not in ['CHI', 'BSE'] and (not self.converged) and (self.parameters.get('IBRION') != 0):
            msg = f'{filename} is an unconverged VASP run.\n'
            msg += f'Electronic convergence reached: {self.converged_electronic}.\n'
            msg += f'Ionic convergence reached: {self.converged_ionic}.'
            warnings.warn(msg, UnconvergedVASPWarning)

    def _parse(self, stream, parse_dos, parse_eigen, parse_projected_eigen):
        self.efermi = self.eigenvalues = self.projected_eigenvalues = self.projected_magnetisation = None
        self.dielectric_data = {}
        self.other_dielectric = {}
        self.incar = {}
        self.kpoints_opt_props = None
        ionic_steps = []
        md_data = []
        parsed_header = False
        in_kpoints_opt = False
        try:
            for event, elem in ET.iterparse(stream, events=['start', 'end']):
                tag = elem.tag
                if event == 'start':
                    if tag == 'calculation':
                        parsed_header = True
                    elif tag in ('eigenvalues_kpoints_opt', 'projected_kpoints_opt'):
                        in_kpoints_opt = True
                else:
                    if not parsed_header:
                        if tag == 'generator':
                            self.generator = self._parse_params(elem)
                        elif tag == 'incar':
                            self.incar = self._parse_params(elem)
                        elif tag == 'kpoints':
                            if not hasattr(self, 'kpoints'):
                                self.kpoints, self.actual_kpoints, self.actual_kpoints_weights = self._parse_kpoints(elem)
                        elif tag == 'parameters':
                            self.parameters = self._parse_params(elem)
                        elif tag == 'structure' and elem.attrib.get('name') == 'initialpos':
                            self.initial_structure = self._parse_structure(elem)
                            self.final_structure = self.initial_structure
                        elif tag == 'atominfo':
                            self.atomic_symbols, self.potcar_symbols = self._parse_atominfo(elem)
                            self.potcar_spec = [{'titel': p, 'hash': None, 'summary_stats': {}} for p in self.potcar_symbols]
                    if tag == 'calculation':
                        parsed_header = True
                        if not self.parameters.get('LCHIMAG', False):
                            ionic_steps.append(self._parse_calculation(elem))
                        else:
                            ionic_steps.extend(self._parse_chemical_shielding_calculation(elem))
                    elif parse_dos and tag == 'dos':
                        if elem.get('comment') == 'kpoints_opt':
                            if self.kpoints_opt_props is None:
                                self.kpoints_opt_props = KpointOptProps()
                            try:
                                self.kpoints_opt_props.tdos, self.kpoints_opt_props.idos, self.kpoints_opt_props.pdos = self._parse_dos(elem)
                                self.kpoints_opt_props.efermi = self.kpoints_opt_props.tdos.efermi
                                self.kpoints_opt_props.dos_has_errors = False
                            except Exception:
                                self.kpoints_opt_props.dos_has_errors = True
                        else:
                            try:
                                self.tdos, self.idos, self.pdos = self._parse_dos(elem)
                                self.efermi = self.tdos.efermi
                                self.dos_has_errors = False
                            except Exception:
                                self.dos_has_errors = True
                    elif parse_eigen and tag == 'eigenvalues' and (not in_kpoints_opt):
                        self.eigenvalues = self._parse_eigen(elem)
                    elif parse_projected_eigen and tag == 'projected' and (not in_kpoints_opt):
                        self.projected_eigenvalues, self.projected_magnetisation = self._parse_projected_eigen(elem)
                    elif tag in ('eigenvalues_kpoints_opt', 'projected_kpoints_opt'):
                        in_kpoints_opt = False
                        if self.kpoints_opt_props is None:
                            self.kpoints_opt_props = KpointOptProps()
                        if parse_eigen:
                            self.kpoints_opt_props.eigenvalues = self._parse_eigen(elem.find('eigenvalues'))
                        if tag == 'eigenvalues_kpoints_opt':
                            self.kpoints_opt_props.kpoints, self.kpoints_opt_props.actual_kpoints, self.kpoints_opt_props.actual_kpoints_weights = self._parse_kpoints(elem.find('kpoints'))
                        elif parse_projected_eigen:
                            self.kpoints_opt_props.projected_eigenvalues, self.kpoints_opt_props.projected_magnetisation = self._parse_projected_eigen(elem)
                    elif tag == 'dielectricfunction':
                        if 'comment' not in elem.attrib or elem.attrib['comment'] == 'INVERSE MACROSCOPIC DIELECTRIC TENSOR (including local field effects in RPA (Hartree))':
                            if 'density' not in self.dielectric_data:
                                self.dielectric_data['density'] = self._parse_diel(elem)
                            elif 'velocity' not in self.dielectric_data:
                                self.dielectric_data['velocity'] = self._parse_diel(elem)
                            else:
                                raise NotImplementedError('This vasprun.xml has >2 unlabelled dielectric functions')
                        else:
                            comment = elem.attrib['comment']
                            if comment == 'density-density':
                                self.dielectric_data['density'] = self._parse_diel(elem)
                            elif comment == 'current-current':
                                self.dielectric_data['velocity'] = self._parse_diel(elem)
                            else:
                                self.other_dielectric[comment] = self._parse_diel(elem)
                    elif tag == 'varray' and elem.attrib.get('name') == 'opticaltransitions':
                        self.optical_transition = np.array(_parse_vasp_array(elem))
                    elif tag == 'structure' and elem.attrib.get('name') == 'finalpos':
                        self.final_structure = self._parse_structure(elem)
                    elif tag == 'dynmat':
                        hessian, eigenvalues, eigenvectors = self._parse_dynmat(elem)
                        n_atoms = len(hessian) // 3
                        hessian = np.array(hessian)
                        self.force_constants = np.zeros((n_atoms, n_atoms, 3, 3), dtype='double')
                        for ii in range(n_atoms):
                            for jj in range(n_atoms):
                                self.force_constants[ii, jj] = hessian[ii * 3:(ii + 1) * 3, jj * 3:(jj + 1) * 3]
                        phonon_eigenvectors = []
                        for ev in eigenvectors:
                            phonon_eigenvectors.append(np.array(ev).reshape(n_atoms, 3))
                        self.normalmode_eigenvals = np.array(eigenvalues)
                        self.normalmode_eigenvecs = np.array(phonon_eigenvectors)
                    elif self.incar.get('ML_LMLFF'):
                        if tag == 'structure' and elem.attrib.get('name') is None:
                            md_data.append({})
                            md_data[-1]['structure'] = self._parse_structure(elem)
                        elif tag == 'varray' and elem.attrib.get('name') == 'forces':
                            md_data[-1]['forces'] = _parse_vasp_array(elem)
                        elif tag == 'varray' and elem.attrib.get('name') == 'stress':
                            md_data[-1]['stress'] = _parse_vasp_array(elem)
                        elif tag == 'energy':
                            d = {i.attrib['name']: float(i.text) for i in elem.findall('i')}
                            if 'kinetic' in d:
                                md_data[-1]['energy'] = {i.attrib['name']: float(i.text) for i in elem.findall('i')}
        except ET.ParseError as exc:
            if self.exception_on_bad_xml:
                raise exc
            warnings.warn('XML is malformed. Parsing has stopped but partial data is available.', UserWarning)
        self.ionic_steps = ionic_steps
        self.md_data = md_data
        self.vasp_version = self.generator['version']

    @property
    def structures(self) -> list[Structure]:
        """
        Returns:
            List of Structure objects for the structure at each ionic step.
        """
        return [step['structure'] for step in self.ionic_steps]

    @property
    def epsilon_static(self) -> list[float]:
        """
        Property only available for DFPT calculations.

        Returns:
            The static part of the dielectric constant. Present when it's a DFPT run
            (LEPSILON=TRUE)
        """
        return self.ionic_steps[-1].get('epsilon', [])

    @property
    def epsilon_static_wolfe(self) -> list[float]:
        """
        Property only available for DFPT calculations.

        Returns:
            The static part of the dielectric constant without any local field
            effects. Present when it's a DFPT run (LEPSILON=TRUE)
        """
        return self.ionic_steps[-1].get('epsilon_rpa', [])

    @property
    def epsilon_ionic(self) -> list[float]:
        """
        Property only available for DFPT calculations and when IBRION=5, 6, 7 or 8.

        Returns:
            The ionic part of the static dielectric constant. Present when it's a
            DFPT run (LEPSILON=TRUE) and IBRION=5, 6, 7 or 8
        """
        return self.ionic_steps[-1].get('epsilon_ion', [])

    @property
    def dielectric(self):
        """
        Returns:
            The real and imaginary part of the dielectric constant (e.g., computed
            by RPA) in function of the energy (frequency). Optical properties (e.g.
            absorption coefficient) can be obtained through this.
            The data is given as a tuple of 3 values containing each of them
            the energy, the real part tensor, and the imaginary part tensor
            ([energies],[[real_partxx,real_partyy,real_partzz,real_partxy,
            real_partyz,real_partxz]],[[imag_partxx,imag_partyy,imag_partzz,
            imag_partxy, imag_partyz, imag_partxz]]).
        """
        return self.dielectric_data['density']

    @property
    def optical_absorption_coeff(self) -> list[float]:
        """
        Calculate the optical absorption coefficient
        from the dielectric constants. Note that this method is only
        implemented for optical properties calculated with GGA and BSE.

        Returns:
            list[float]: optical absorption coefficient
        """
        if self.dielectric_data['density']:
            real_avg = [sum(self.dielectric_data['density'][1][i][0:3]) / 3 for i in range(len(self.dielectric_data['density'][0]))]
            imag_avg = [sum(self.dielectric_data['density'][2][i][0:3]) / 3 for i in range(len(self.dielectric_data['density'][0]))]

            def optical_absorb_coeff(freq, real, imag):
                """
                The optical absorption coefficient calculated in terms of
                equation, the unit is cm^-1.
                """
                hc = 1.23984 * 0.0001
                return 2 * 3.14159 * np.sqrt(np.sqrt(real ** 2 + imag ** 2) - real) * np.sqrt(2) / hc * freq
            absorption_coeff = list(itertools.starmap(optical_absorb_coeff, zip(self.dielectric_data['density'][0], real_avg, imag_avg)))
        return absorption_coeff

    @property
    def converged_electronic(self) -> bool:
        """
        Returns:
            bool: True if electronic step convergence has been reached in the final ionic step.
        """
        final_elec_steps = self.ionic_steps[-1]['electronic_steps'] if self.incar.get('ALGO', '').lower() != 'chi' else 0
        if self.incar.get('LEPSILON'):
            idx = 1
            to_check = {'e_wo_entrp', 'e_fr_energy', 'e_0_energy'}
            while set(final_elec_steps[idx]) == to_check:
                idx += 1
            return idx + 1 != self.parameters['NELM']
        return len(final_elec_steps) < self.parameters['NELM']

    @property
    def converged_ionic(self) -> bool:
        """
        Returns:
            bool: True if ionic step convergence has been reached, i.e. that vasp
                exited before reaching the max ionic steps for a relaxation run.
                In case IBRION=0 (MD) True if the max ionic steps are reached.
        """
        nsw = self.parameters.get('NSW', 0)
        ibrion = self.parameters.get('IBRION', -1 if nsw in (-1, 0) else 0)
        if ibrion == 0:
            return nsw <= 1 or self.md_n_steps == nsw
        return nsw <= 1 or len(self.ionic_steps) < nsw

    @property
    def converged(self) -> bool:
        """
        Returns:
            bool: True if a relaxation run is both ionically and electronically converged.
        """
        return self.converged_electronic and self.converged_ionic

    @property
    @unitized('eV')
    def final_energy(self):
        """Final energy from the vasp run."""
        try:
            final_istep = self.ionic_steps[-1]
            total_energy = final_istep['e_0_energy']
            final_estep = final_istep['electronic_steps'][-1]
            electronic_energy_diff = final_estep['e_0_energy'] - final_estep['e_fr_energy']
            total_energy_bugfix = np.round(electronic_energy_diff + final_istep['e_fr_energy'], 8)
            if np.abs(total_energy - total_energy_bugfix) > 1e-07:
                return total_energy_bugfix
            return total_energy
        except (IndexError, KeyError):
            warnings.warn('Calculation does not have a total energy. Possibly a GW or similar kind of run. A value of infinity is returned.')
            return float('inf')

    @property
    def complete_dos(self):
        """
        A complete dos object which incorporates the total dos and all
        projected dos.
        """
        final_struct = self.final_structure
        pdoss = {final_struct[i]: pdos for i, pdos in enumerate(self.pdos)}
        return CompleteDos(self.final_structure, self.tdos, pdoss)

    @property
    def complete_dos_normalized(self) -> CompleteDos:
        """
        A CompleteDos object which incorporates the total DOS and all projected DOS.
        Normalized by the volume of the unit cell with units of states/eV/unit cell
        volume.
        """
        final_struct = self.final_structure
        pdoss = {final_struct[i]: pdos for i, pdos in enumerate(self.pdos)}
        return CompleteDos(self.final_structure, self.tdos, pdoss, normalize=True)

    @property
    def hubbards(self) -> dict[str, float]:
        """Hubbard U values used if a vasprun is a GGA+U run. Otherwise an empty dict."""
        symbols = [s.split()[1] for s in self.potcar_symbols]
        symbols = [re.split('_', s)[0] for s in symbols]
        if not self.incar.get('LDAU', False):
            return {}
        us = self.incar.get('LDAUU', self.parameters.get('LDAUU'))
        js = self.incar.get('LDAUJ', self.parameters.get('LDAUJ'))
        if len(js) != len(us):
            js = [0] * len(us)
        if len(us) == len(symbols):
            return {symbols[idx]: us[idx] - js[idx] for idx in range(len(symbols))}
        if sum(us) == 0 and sum(js) == 0:
            return {}
        raise VaspParseError('Length of U value parameters and atomic symbols are mismatched')

    @property
    def run_type(self):
        """
        Returns the run type. Currently detects GGA, metaGGA, HF, HSE, B3LYP,
        and hybrid functionals based on relevant INCAR tags. LDA is assigned if
        PAW POTCARs are used and no other functional is detected.

        Hubbard U terms and vdW corrections are detected automatically as well.
        """
        GGA_TYPES = {'RE': 'revPBE', 'PE': 'PBE', 'PS': 'PBEsol', 'RP': 'revPBE+Padé', 'AM': 'AM05', 'OR': 'optPBE', 'BO': 'optB88', 'MK': 'optB86b', '--': 'GGA'}
        METAGGA_TYPES = {'TPSS': 'TPSS', 'RTPSS': 'revTPSS', 'M06L': 'M06-L', 'MBJ': 'modified Becke-Johnson', 'SCAN': 'SCAN', 'R2SCAN': 'R2SCAN', 'RSCAN': 'RSCAN', 'MS0': 'MadeSimple0', 'MS1': 'MadeSimple1', 'MS2': 'MadeSimple2'}
        IVDW_TYPES = {1: 'DFT-D2', 10: 'DFT-D2', 11: 'DFT-D3', 12: 'DFT-D3-BJ', 2: 'TS', 20: 'TS', 21: 'TS-H', 202: 'MBD', 4: 'dDsC'}
        if self.parameters.get('AEXX', 1.0) == 1.0:
            rt = 'HF'
        elif self.parameters.get('HFSCREEN', 0.3) == 0.3:
            rt = 'HSE03'
        elif self.parameters.get('HFSCREEN', 0.2) == 0.2:
            rt = 'HSE06'
        elif self.parameters.get('AEXX', 0.2) == 0.2:
            rt = 'B3LYP'
        elif self.parameters.get('LHFCALC', True):
            rt = 'PBEO or other Hybrid Functional'
        elif self.incar.get('METAGGA') and self.incar.get('METAGGA') not in ['--', 'None']:
            incar_tag = self.incar.get('METAGGA', '').strip().upper()
            rt = METAGGA_TYPES.get(incar_tag, incar_tag)
        elif self.parameters.get('GGA'):
            incar_tag = self.parameters.get('GGA', '').strip().upper()
            rt = GGA_TYPES.get(incar_tag, incar_tag)
        elif self.potcar_symbols[0].split()[0] == 'PAW':
            rt = 'LDA'
        else:
            rt = 'unknown'
            warnings.warn('Unknown run type!')
        if self.is_hubbard or self.parameters.get('LDAU', True):
            rt += '+U'
        if self.parameters.get('LUSE_VDW', False):
            rt += '+rVV10'
        elif self.incar.get('IVDW') in IVDW_TYPES:
            rt += '+vdW-' + IVDW_TYPES[self.incar.get('IVDW')]
        elif self.incar.get('IVDW'):
            rt += '+vdW-unknown'
        return rt

    @property
    def is_hubbard(self) -> bool:
        """True if run is a DFT+U run."""
        if len(self.hubbards) == 0:
            return False
        return sum(self.hubbards.values()) > 1e-08

    @property
    def is_spin(self) -> bool:
        """True if run is spin-polarized."""
        return self.parameters.get('ISPIN', 1) == 2

    @property
    def md_n_steps(self) -> int:
        """Number of steps for md runs."""
        if self.md_data:
            return len(self.md_data)
        return self.nionic_steps

    def get_computed_entry(self, inc_structure=True, parameters=None, data=None, entry_id: str | None=None):
        """
        Returns a ComputedEntry or ComputedStructureEntry from the Vasprun.

        Args:
            inc_structure (bool): Set to True if you want
                ComputedStructureEntries to be returned instead of
                ComputedEntries.
            parameters (list): Input parameters to include. It has to be one of
                the properties supported by the Vasprun object. If
                parameters is None, a default set of parameters that are
                necessary for typical post-processing will be set.
            data (list): Output data to include. Has to be one of the properties
                supported by the Vasprun object.
            entry_id (str): Specify an entry id for the ComputedEntry. Defaults to
                "vasprun-{current datetime}"

        Returns:
            ComputedStructureEntry/ComputedEntry
        """
        if entry_id is None:
            entry_id = f'vasprun-{datetime.datetime.now()}'
        param_names = {'is_hubbard', 'hubbards', 'potcar_symbols', 'potcar_spec', 'run_type'}
        if parameters:
            param_names.update(parameters)
        params = {p: getattr(self, p) for p in param_names}
        data = {p: getattr(self, p) for p in data} if data is not None else {}
        if inc_structure:
            return ComputedStructureEntry(self.final_structure, self.final_energy, parameters=params, data=data, entry_id=entry_id)
        return ComputedEntry(self.final_structure.composition, self.final_energy, parameters=params, data=data, entry_id=entry_id)

    def get_band_structure(self, kpoints_filename: str | None=None, efermi: float | Literal['smart'] | None=None, line_mode: bool=False, force_hybrid_mode: bool=False, ignore_kpoints_opt: bool=False) -> BandStructureSymmLine | BandStructure:
        """Get the band structure as a BandStructure object.

        Args:
            kpoints_filename: Full path of the KPOINTS file from which
                the band structure is generated.
                If none is provided, the code will try to intelligently
                determine the appropriate KPOINTS file by substituting the
                filename of the vasprun.xml with KPOINTS (or KPOINTS_OPT).
                The latter is the default behavior.
            efermi: The Fermi energy associated with the bandstructure, in eV. By
                default (None), uses the value reported by VASP in vasprun.xml. To
                manually set the Fermi energy, pass a float. Pass 'smart' to use the
                `calculate_efermi()` method, which calculates the Fermi level by first
                checking whether it lies within a small tolerance (by default 0.001 eV)
                of a band edge) If it does, the Fermi level is placed in the center of
                the bandgap. Otherwise, the value is identical to the value reported by
                VASP.
            line_mode: Force the band structure to be considered as
                a run along symmetry lines. (Default: False)
            force_hybrid_mode: Makes it possible to read in self-consistent band
                structure calculations for every type of functional. (Default: False)
            ignore_kpoints_opt: Normally, if KPOINTS_OPT data exists, it has
                the band structure data. Set this flag to ignore it. (Default: False)

        Returns:
            a BandStructure object (or more specifically a
            BandStructureSymmLine object if the run is detected to be a run
            along symmetry lines)

            Two types of runs along symmetry lines are accepted: non-sc with
            Line-Mode in the KPOINT file or hybrid, self-consistent with a
            uniform grid+a few kpoints along symmetry lines (explicit KPOINTS
            file) (it's not possible to run a non-sc band structure with hybrid
            functionals). The explicit KPOINTS file needs to have data on the
            kpoint label as commentary.

            If VASP was run with KPOINTS_OPT, it reads the data from that
            file unless told otherwise. This overrides hybrid mode.
        """
        use_kpoints_opt = not ignore_kpoints_opt and getattr(self, 'kpoints_opt_props', None) is not None
        if not kpoints_filename:
            kpts_path = os.path.join(os.path.dirname(self.filename), 'KPOINTS_OPT' if use_kpoints_opt else 'KPOINTS')
            kpoints_filename = zpath(kpts_path)
        if kpoints_filename and (not os.path.isfile(kpoints_filename)) and (line_mode is True):
            name = 'KPOINTS_OPT' if use_kpoints_opt else 'KPOINTS'
            raise VaspParseError(f'{name} not found but needed to obtain band structure along symmetry lines.')
        if efermi == 'smart':
            e_fermi = self.calculate_efermi()
        elif efermi is None:
            e_fermi = self.efermi
        else:
            e_fermi = efermi
        kpoint_file: Kpoints = None
        if kpoints_filename and os.path.isfile(kpoints_filename):
            kpoint_file = Kpoints.from_file(kpoints_filename)
        lattice_new = Lattice(self.final_structure.lattice.reciprocal_lattice.matrix)
        if use_kpoints_opt:
            kpoints = [np.array(kpt) for kpt in self.kpoints_opt_props.actual_kpoints]
        else:
            kpoints = [np.array(kpt) for kpt in self.actual_kpoints]
        p_eig_vals: defaultdict[Spin, list] = defaultdict(list)
        eigenvals: defaultdict[Spin, list] = defaultdict(list)
        n_kpts = len(kpoints)
        if use_kpoints_opt:
            eig_vals = self.kpoints_opt_props.eigenvalues
            projected_eig_vals = self.kpoints_opt_props.projected_eigenvalues
        else:
            eig_vals = self.eigenvalues
            projected_eig_vals = self.projected_eigenvalues
        for spin, val in eig_vals.items():
            val = np.swapaxes(val, 0, 1)
            eigenvals[spin] = val[:, :, 0]
            if projected_eig_vals:
                proj_eig_vals = projected_eig_vals[spin]
                proj_eig_vals = np.swapaxes(proj_eig_vals, 0, 1)
                proj_eig_vals = np.swapaxes(proj_eig_vals, 2, 3)
                p_eig_vals[spin] = proj_eig_vals
        hybrid_band = False
        if self.parameters.get('LHFCALC', False) or 0.0 in self.actual_kpoints_weights:
            hybrid_band = True
        if kpoint_file is not None and kpoint_file.style == Kpoints.supported_modes.Line_mode:
            line_mode = True
        if line_mode:
            labels_dict = {}
            if (hybrid_band or force_hybrid_mode) and (not use_kpoints_opt):
                start_bs_index = 0
                for i in range(len(self.actual_kpoints)):
                    if self.actual_kpoints_weights[i] == 0.0:
                        start_bs_index = i
                        break
                for i in range(start_bs_index, len(kpoint_file.kpts)):
                    if kpoint_file.labels[i] is not None:
                        labels_dict[kpoint_file.labels[i]] = kpoint_file.kpts[i]
                n_bands = len(eigenvals[Spin.up])
                kpoints = kpoints[start_bs_index:n_kpts]
                up_eigen = [eigenvals[Spin.up][i][start_bs_index:n_kpts] for i in range(n_bands)]
                if self.projected_eigenvalues:
                    p_eig_vals[Spin.up] = [p_eig_vals[Spin.up][i][start_bs_index:n_kpts] for i in range(n_bands)]
                if self.is_spin:
                    down_eigen = [eigenvals[Spin.down][i][start_bs_index:n_kpts] for i in range(n_bands)]
                    eigenvals[Spin.up] = up_eigen
                    eigenvals[Spin.down] = down_eigen
                    if self.projected_eigenvalues:
                        p_eig_vals[Spin.down] = [p_eig_vals[Spin.down][i][start_bs_index:n_kpts] for i in range(n_bands)]
                else:
                    eigenvals[Spin.up] = up_eigen
            else:
                if '' in kpoint_file.labels:
                    raise ValueError('A band structure along symmetry lines requires a label for each kpoint. Check your KPOINTS file')
                labels_dict = dict(zip(kpoint_file.labels, kpoint_file.kpts))
                labels_dict.pop(None, None)
            return BandStructureSymmLine(kpoints, eigenvals, lattice_new, e_fermi, labels_dict, structure=self.final_structure, projections=p_eig_vals)
        return BandStructure(kpoints, eigenvals, lattice_new, e_fermi, structure=self.final_structure, projections=p_eig_vals)

    @property
    def eigenvalue_band_properties(self):
        """
        Band properties from the eigenvalues as a tuple,
        (band gap, cbm, vbm, is_band_gap_direct). In the case of separate_spins=True,
        the band gap, cbm, vbm, and is_band_gap_direct are each lists of length 2,
        with index 0 representing the spin-up channel and index 1 representing
        the spin-down channel.
        """
        vbm = -float('inf')
        vbm_kpoint = None
        cbm = float('inf')
        cbm_kpoint = None
        vbm_spins = []
        vbm_spins_kpoints = []
        cbm_spins = []
        cbm_spins_kpoints = []
        if self.separate_spins and len(self.eigenvalues) != 2:
            raise ValueError('The separate_spins flag can only be True if ISPIN = 2')
        for d in self.eigenvalues.values():
            if self.separate_spins:
                vbm = -float('inf')
                cbm = float('inf')
            for k, val in enumerate(d):
                for eigenval, occu in val:
                    if occu > self.occu_tol and eigenval > vbm:
                        vbm = eigenval
                        vbm_kpoint = k
                    elif occu <= self.occu_tol and eigenval < cbm:
                        cbm = eigenval
                        cbm_kpoint = k
            if self.separate_spins:
                vbm_spins.append(vbm)
                vbm_spins_kpoints.append(vbm_kpoint)
                cbm_spins.append(cbm)
                cbm_spins_kpoints.append(cbm_kpoint)
        if self.separate_spins:
            return ([max(cbm_spins[0] - vbm_spins[0], 0), max(cbm_spins[1] - vbm_spins[1], 0)], [cbm_spins[0], cbm_spins[1]], [vbm_spins[0], vbm_spins[1]], [vbm_spins_kpoints[0] == cbm_spins_kpoints[0], vbm_spins_kpoints[1] == cbm_spins_kpoints[1]])
        return (max(cbm - vbm, 0), cbm, vbm, vbm_kpoint == cbm_kpoint)

    def calculate_efermi(self, tol: float=0.001):
        """
        Calculate the Fermi level using a robust algorithm.

        Sometimes VASP can put the Fermi level just inside of a band due to issues in
        the way band occupancies are handled. This algorithm tries to detect and correct
        for this bug.

        Slightly more details are provided here: https://www.vasp.at/forum/viewtopic.php?f=4&t=17981
        """
        all_eigs = np.concatenate([eigs[:, :, 0].transpose(1, 0) for eigs in self.eigenvalues.values()])

        def crosses_band(fermi):
            eigs_below = np.any(all_eigs < fermi, axis=1)
            eigs_above = np.any(all_eigs > fermi, axis=1)
            return np.any(eigs_above & eigs_below)

        def get_vbm_cbm(fermi):
            return (np.max(all_eigs[all_eigs < fermi]), np.min(all_eigs[all_eigs > fermi]))
        if not crosses_band(self.efermi):
            return self.efermi
        if not crosses_band(self.efermi + tol):
            vbm, cbm = get_vbm_cbm(self.efermi + tol)
            return (cbm + vbm) / 2
        if not crosses_band(self.efermi - tol):
            vbm, cbm = get_vbm_cbm(self.efermi - tol)
            return (cbm + vbm) / 2
        return self.efermi

    def get_potcars(self, path: str | Path | bool) -> Potcar | None:
        """Returns the POTCAR from the specified path.

        Args:
            path (str | Path | bool): If a str or Path, the path to search for POTCARs.
                If a bool, whether to take the search path from the specified vasprun.xml

        Returns:
            Potcar | None: The POTCAR from the specified path or None if not found/no path specified.
        """
        if not path:
            return None
        if isinstance(path, (str, Path)) and 'POTCAR' in str(path):
            potcar_paths = [str(path)]
        else:
            search_path = os.path.dirname(os.path.abspath(self.filename)) if path is True else str(path)
            potcar_paths = [f'{search_path}/{fn}' for fn in os.listdir(search_path) if fn.startswith('POTCAR') and '.spec' not in fn]
        for potcar_path in potcar_paths:
            try:
                potcar = Potcar.from_file(potcar_path)
                if {d.header for d in potcar} == set(self.potcar_symbols):
                    return potcar
            except Exception:
                continue
        warnings.warn('No POTCAR file with matching TITEL fields was found in\n' + '\n  '.join(potcar_paths))
        return None

    def get_trajectory(self):
        """
        This method returns a Trajectory object, which is an alternative
        representation of self.structures into a single object. Forces are
        added to the Trajectory as site properties.

        Returns:
            Trajectory: from pymatgen.core.trajectory
        """
        from pymatgen.core.trajectory import Trajectory
        structs = []
        steps_list = self.md_data or self.ionic_steps
        for step in steps_list:
            struct = step['structure'].copy()
            struct.add_site_property('forces', step['forces'])
            structs.append(struct)
        return Trajectory.from_structures(structs, constant_lattice=False)

    def update_potcar_spec(self, path):
        """
        Args:
            path: Path to search for POTCARs

        Returns:
            Potcar spec from path.
        """
        if (potcar := self.get_potcars(path)):
            self.potcar_spec = [{'titel': sym, 'hash': ps.md5_header_hash, 'summary_stats': ps._summary_stats} for sym in self.potcar_symbols for ps in potcar if ps.symbol == sym.split()[1]]

    def update_charge_from_potcar(self, path):
        """
        Sets the charge of a structure based on the POTCARs found.

        Args:
            path: Path to search for POTCARs
        """
        potcar = self.get_potcars(path)
        if potcar and self.incar.get('ALGO', '') not in ['GW0', 'G0W0', 'GW', 'BSE']:
            nelect = self.parameters['NELECT']
            if len(potcar) == len(self.initial_structure.composition.element_composition):
                potcar_nelect = sum((self.initial_structure.composition.element_composition[ps.element] * ps.ZVAL for ps in potcar))
            else:
                nums = [len(list(g)) for _, g in itertools.groupby(self.atomic_symbols)]
                potcar_nelect = sum((ps.ZVAL * num for ps, num in zip(potcar, nums)))
            charge = potcar_nelect - nelect
            for s in self.structures:
                s._charge = charge
            if hasattr(self, 'initial_structure'):
                self.initial_structure._charge = charge
            if hasattr(self, 'final_structure'):
                self.final_structure._charge = charge

    def as_dict(self):
        """JSON-serializable dict representation."""
        dct = {'vasp_version': self.vasp_version, 'has_vasp_completed': self.converged, 'nsites': len(self.final_structure)}
        comp = self.final_structure.composition
        dct['unit_cell_formula'] = comp.as_dict()
        dct['reduced_cell_formula'] = Composition(comp.reduced_formula).as_dict()
        dct['pretty_formula'] = comp.reduced_formula
        symbols = [s.split()[1] for s in self.potcar_symbols]
        symbols = [re.split('_', s)[0] for s in symbols]
        dct['is_hubbard'] = self.is_hubbard
        dct['hubbards'] = self.hubbards
        unique_symbols = sorted(set(self.atomic_symbols))
        dct['elements'] = unique_symbols
        dct['nelements'] = len(unique_symbols)
        dct['run_type'] = self.run_type
        vin = {'incar': dict(self.incar.items()), 'crystal': self.initial_structure.as_dict(), 'kpoints': self.kpoints.as_dict()}
        actual_kpts = [{'abc': list(self.actual_kpoints[idx]), 'weight': self.actual_kpoints_weights[idx]} for idx in range(len(self.actual_kpoints))]
        vin['kpoints']['actual_points'] = actual_kpts
        vin['nkpoints'] = len(actual_kpts)
        if (kpt_opt_props := getattr(self, 'kpoints_opt_props', None)):
            vin['kpoints_opt'] = kpt_opt_props.kpoints.as_dict()
            actual_kpts = [{'abc': list(kpt_opt_props.actual_kpoints[idx]), 'weight': kpt_opt_props.actual_kpoints_weights[idx]} for idx in range(len(kpt_opt_props.actual_kpoints))]
            vin['kpoints_opt']['actual_kpoints'] = actual_kpts
            vin['nkpoints_opt'] = len(actual_kpts)
        vin['potcar'] = [s.split(' ')[1] for s in self.potcar_symbols]
        vin['potcar_spec'] = self.potcar_spec
        vin['potcar_type'] = [s.split(' ')[0] for s in self.potcar_symbols]
        vin['parameters'] = dict(self.parameters.items())
        vin['lattice_rec'] = self.final_structure.lattice.reciprocal_lattice.as_dict()
        dct['input'] = vin
        n_sites = len(self.final_structure)
        try:
            vout = {'ionic_steps': self.ionic_steps, 'final_energy': self.final_energy, 'final_energy_per_atom': self.final_energy / n_sites, 'crystal': self.final_structure.as_dict(), 'efermi': self.efermi}
        except (ArithmeticError, TypeError):
            vout = {'ionic_steps': self.ionic_steps, 'final_energy': self.final_energy, 'final_energy_per_atom': None, 'crystal': self.final_structure.as_dict(), 'efermi': self.efermi}
        if self.eigenvalues:
            eigen = {str(spin): v.tolist() for spin, v in self.eigenvalues.items()}
            vout['eigenvalues'] = eigen
            gap, cbm, vbm, is_direct = self.eigenvalue_band_properties
            vout.update({'bandgap': gap, 'cbm': cbm, 'vbm': vbm, 'is_gap_direct': is_direct})
            if self.projected_eigenvalues:
                vout['projected_eigenvalues'] = {str(spin): v.tolist() for spin, v in self.projected_eigenvalues.items()}
            if self.projected_magnetisation is not None:
                vout['projected_magnetisation'] = self.projected_magnetisation.tolist()
        if kpt_opt_props and kpt_opt_props.eigenvalues:
            eigen = {str(spin): v.tolist() for spin, v in kpt_opt_props.eigenvalues.items()}
            vout['eigenvalues_kpoints_opt'] = eigen
            if kpt_opt_props.projected_eigenvalues:
                vout['projected_eigenvalues_kpoints_opt'] = {str(spin): v.tolist() for spin, v in kpt_opt_props.projected_eigenvalues.items()}
            if kpt_opt_props.projected_magnetisation is not None:
                vout['projected_magnetisation_kpoints_opt'] = kpt_opt_props.projected_magnetisation.tolist()
        vout['epsilon_static'] = self.epsilon_static
        vout['epsilon_static_wolfe'] = self.epsilon_static_wolfe
        vout['epsilon_ionic'] = self.epsilon_ionic
        dct['output'] = vout
        return jsanitize(dct, strict=True)

    def _parse_params(self, elem):
        params = {}
        for c in elem:
            name = c.attrib.get('name')
            if c.tag not in ('i', 'v'):
                p = self._parse_params(c)
                if name == 'response functions':
                    p = {k: v for k, v in p.items() if k not in params}
                params.update(p)
            else:
                ptype = c.attrib.get('type')
                val = c.text.strip() if c.text else ''
                try:
                    if c.tag == 'i':
                        params[name] = _parse_parameters(ptype, val)
                    else:
                        params[name] = _parse_v_parameters(ptype, val, self.filename, name)
                except Exception as exc:
                    if name == 'RANDOM_SEED':
                        params[name] = None
                    else:
                        raise exc
        elem.clear()
        return Incar(params)

    @staticmethod
    def _parse_atominfo(elem):
        for a in elem.findall('array'):
            if a.attrib['name'] == 'atoms':
                atomic_symbols = [rc.find('c').text.strip() for rc in a.find('set')]
            elif a.attrib['name'] == 'atomtypes':
                potcar_symbols = [rc.findall('c')[4].text.strip() for rc in a.find('set')]

        def parse_atomic_symbol(symbol):
            try:
                return str(Element(symbol))
            except ValueError as exc:
                if symbol == 'X':
                    return 'Xe'
                if symbol == 'r':
                    return 'Zr'
                raise exc
        elem.clear()
        return ([parse_atomic_symbol(sym) for sym in atomic_symbols], potcar_symbols)

    @staticmethod
    def _parse_kpoints(elem):
        e = elem
        if elem.find('generation'):
            e = elem.find('generation')
        k = Kpoints('Kpoints from vasprun.xml')
        k.style = Kpoints.supported_modes.from_str(e.attrib.get('param', 'Reciprocal'))
        for v in e.findall('v'):
            name = v.attrib.get('name')
            tokens = v.text.split()
            if name == 'divisions':
                k.kpts = [[int(i) for i in tokens]]
            elif name == 'usershift':
                k.kpts_shift = [float(i) for i in tokens]
            elif name in {'genvec1', 'genvec2', 'genvec3', 'shift'}:
                setattr(k, name, [float(i) for i in tokens])
        for va in elem.findall('varray'):
            name = va.attrib['name']
            if name == 'kpointlist':
                actual_kpoints = _parse_vasp_array(va)
            elif name == 'weights':
                weights = [i[0] for i in _parse_vasp_array(va)]
        elem.clear()
        if k.style == Kpoints.supported_modes.Reciprocal:
            k = Kpoints(comment='Kpoints from vasprun.xml', style=Kpoints.supported_modes.Reciprocal, num_kpts=len(k.kpts), kpts=actual_kpoints, kpts_weights=weights)
        return (k, actual_kpoints, weights)

    def _parse_structure(self, elem):
        lattice = _parse_vasp_array(elem.find('crystal').find('varray'))
        pos = _parse_vasp_array(elem.find('varray'))
        struct = Structure(lattice, self.atomic_symbols, pos)
        selective_dyn = elem.find("varray/[@name='selective']")
        if selective_dyn:
            struct.add_site_property('selective_dynamics', _parse_vasp_array(selective_dyn))
        return struct

    @staticmethod
    def _parse_diel(elem):
        imag = [[_vasprun_float(line) for line in r.text.split()] for r in elem.find('imag').find('array').find('set').findall('r')]
        real = [[_vasprun_float(line) for line in r.text.split()] for r in elem.find('real').find('array').find('set').findall('r')]
        elem.clear()
        return ([e[0] for e in imag], [e[1:] for e in real], [e[1:] for e in imag])

    @staticmethod
    def _parse_optical_transition(elem):
        for va in elem.findall('varray'):
            if va.attrib.get('name') == 'opticaltransitions':
                oscillator_strength = np.array(_parse_vasp_array(va))[0:]
                probability_transition = np.array(_parse_vasp_array(va))[0:, 1]
        return (oscillator_strength, probability_transition)

    def _parse_chemical_shielding_calculation(self, elem):
        calculation = []
        istep = {}
        try:
            struct = self._parse_structure(elem.find('structure'))
        except AttributeError:
            struct = None
        for va in elem.findall('varray'):
            istep[va.attrib['name']] = _parse_vasp_array(va)
        istep['structure'] = struct
        istep['electronic_steps'] = []
        calculation.append(istep)
        for scstep in elem.findall('scstep'):
            try:
                e_steps_dict = {i.attrib['name']: _vasprun_float(i.text) for i in scstep.find('energy').findall('i')}
                cur_ene = e_steps_dict['e_fr_energy']
                min_steps = 1 if len(calculation) >= 1 else self.parameters.get('NELMIN', 5)
                if len(calculation[-1]['electronic_steps']) <= min_steps:
                    calculation[-1]['electronic_steps'].append(e_steps_dict)
                else:
                    last_ene = calculation[-1]['electronic_steps'][-1]['e_fr_energy']
                    if abs(cur_ene - last_ene) < 1.0:
                        calculation[-1]['electronic_steps'].append(e_steps_dict)
                    else:
                        calculation.append({'electronic_steps': [e_steps_dict]})
            except AttributeError:
                pass
        calculation[-1].update(calculation[-1]['electronic_steps'][-1])
        return calculation

    def _parse_calculation(self, elem):
        try:
            istep = {i.attrib['name']: _vasprun_float(i.text) for i in elem.find('energy').findall('i')}
        except AttributeError:
            istep = {}
        esteps = []
        for scstep in elem.findall('scstep'):
            try:
                e_step_dict = {i.attrib['name']: _vasprun_float(i.text) for i in scstep.find('energy').findall('i')}
                esteps.append(e_step_dict)
            except AttributeError:
                pass
        try:
            struct = self._parse_structure(elem.find('structure'))
        except AttributeError:
            struct = None
        for va in elem.findall('varray'):
            istep[va.attrib['name']] = _parse_vasp_array(va)
        istep['electronic_steps'] = esteps
        istep['structure'] = struct
        elem.clear()
        return istep

    @staticmethod
    def _parse_dos(elem):
        efermi = float(elem.find('i').text)
        energies = None
        tdensities = {}
        idensities = {}
        for s in elem.find('total').find('array').find('set').findall('set'):
            data = np.array(_parse_vasp_array(s))
            energies = data[:, 0]
            spin = Spin.up if s.attrib['comment'] == 'spin 1' else Spin.down
            tdensities[spin] = data[:, 1]
            idensities[spin] = data[:, 2]
        pdoss = []
        partial = elem.find('partial')
        if partial is not None:
            orbs = [ss.text for ss in partial.find('array').findall('field')]
            orbs.pop(0)
            lm = any(('x' in s for s in orbs))
            for s in partial.find('array').find('set').findall('set'):
                pdos = defaultdict(dict)
                for ss in s.findall('set'):
                    spin = Spin.up if ss.attrib['comment'] == 'spin 1' else Spin.down
                    data = np.array(_parse_vasp_array(ss))
                    _nrow, ncol = data.shape
                    for j in range(1, ncol):
                        orb = Orbital(j - 1) if lm else OrbitalType(j - 1)
                        pdos[orb][spin] = data[:, j]
                pdoss.append(pdos)
        elem.clear()
        return (Dos(efermi, energies, tdensities), Dos(efermi, energies, idensities), pdoss)

    @staticmethod
    def _parse_eigen(elem):
        eigenvalues = defaultdict(list)
        for s in elem.find('array').find('set').findall('set'):
            spin = Spin.up if s.attrib['comment'] == 'spin 1' else Spin.down
            for ss in s.findall('set'):
                eigenvalues[spin].append(_parse_vasp_array(ss))
        eigenvalues = {spin: np.array(v) for spin, v in eigenvalues.items()}
        elem.clear()
        return eigenvalues

    @staticmethod
    def _parse_projected_eigen(elem):
        root = elem.find('array').find('set')
        proj_eigen = defaultdict(list)
        for s in root.findall('set'):
            spin = int(re.match('spin(\\d+)', s.attrib['comment']).group(1))
            for ss in s.findall('set'):
                dk = []
                for sss in ss.findall('set'):
                    db = _parse_vasp_array(sss)
                    dk.append(db)
                proj_eigen[spin].append(dk)
        proj_eigen = {spin: np.array(v) for spin, v in proj_eigen.items()}
        if len(proj_eigen) > 2:
            proj_mag = np.stack([proj_eigen.pop(i) for i in range(2, 5)], axis=-1)
            proj_eigen = {Spin.up: proj_eigen[1]}
        else:
            proj_eigen = {Spin.up if k == 1 else Spin.down: v for k, v in proj_eigen.items()}
            proj_mag = None
        elem.clear()
        return (proj_eigen, proj_mag)

    @staticmethod
    def _parse_dynmat(elem):
        hessian = []
        eigenvalues = []
        eigenvectors = []
        for v in elem.findall('v'):
            if v.attrib['name'] == 'eigenvalues':
                eigenvalues = [float(i) for i in v.text.split()]
        for va in elem.findall('varray'):
            if va.attrib['name'] == 'hessian':
                for v in va.findall('v'):
                    hessian.append([float(i) for i in v.text.split()])
            elif va.attrib['name'] == 'eigenvectors':
                for v in va.findall('v'):
                    eigenvectors.append([float(i) for i in v.text.split()])
        return (hessian, eigenvalues, eigenvectors)