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
class BSVasprun(Vasprun):
    """
    A highly optimized version of Vasprun that parses only eigenvalues for
    bandstructures. All other properties like structures, parameters,
    etc. are ignored.
    """

    def __init__(self, filename: str, parse_projected_eigen: bool | str=False, parse_potcar_file: bool | str=False, occu_tol: float=1e-08, separate_spins: bool=False):
        """
        Args:
            filename: Filename to parse
            parse_projected_eigen: Whether to parse the projected
                eigenvalues. Defaults to False. Set to True to obtain projected
                eigenvalues. **Note that this can take an extreme amount of time
                and memory.** So use this wisely.
            parse_potcar_file: Whether to parse the potcar file to read
                the potcar hashes for the potcar_spec attribute. Defaults to True,
                where no hashes will be determined and the potcar_spec dictionaries
                will read {"symbol": ElSymbol, "hash": None}. By Default, looks in
                the same directory as the vasprun.xml, with same extensions as
                Vasprun.xml. If a string is provided, looks at that filepath.
            occu_tol: Sets the minimum tol for the determination of the
                vbm and cbm. Usually the default of 1e-8 works well enough,
                but there may be pathological cases.
            separate_spins (bool): Whether the band gap, CBM, and VBM should be
                reported for each individual spin channel. Defaults to False,
                which computes the eigenvalue band properties independent of
                the spin orientation. If True, the calculation must be spin-polarized.
        """
        self.filename = filename
        self.occu_tol = occu_tol
        self.separate_spins = separate_spins
        with zopen(filename, mode='rt') as file:
            self.efermi = None
            parsed_header = False
            in_kpoints_opt = False
            self.eigenvalues = self.projected_eigenvalues = None
            self.kpoints_opt_props = None
            for event, elem in ET.iterparse(file, events=['start', 'end']):
                tag = elem.tag
                if event == 'start':
                    if tag == 'eigenvalues_kpoints_opt' or tag == 'projected_kpoints_opt' or (tag == 'dos' and elem.attrib.get('comment') == 'kpoints_opt'):
                        in_kpoints_opt = True
                elif not parsed_header:
                    if tag == 'generator':
                        self.generator = self._parse_params(elem)
                    elif tag == 'incar':
                        self.incar = self._parse_params(elem)
                    elif tag == 'kpoints':
                        self.kpoints, self.actual_kpoints, self.actual_kpoints_weights = self._parse_kpoints(elem)
                    elif tag == 'parameters':
                        self.parameters = self._parse_params(elem)
                    elif tag == 'atominfo':
                        self.atomic_symbols, self.potcar_symbols = self._parse_atominfo(elem)
                        self.potcar_spec = [{'titel': p, 'hash': None, 'summary_stats': {}} for p in self.potcar_symbols]
                        parsed_header = True
                elif tag == 'i' and elem.attrib.get('name') == 'efermi':
                    if in_kpoints_opt:
                        if self.kpoints_opt_props is None:
                            self.kpoints_opt_props = KpointOptProps()
                        self.kpoints_opt_props.efermi = float(elem.text)
                        in_kpoints_opt = False
                    else:
                        self.efermi = float(elem.text)
                elif tag == 'eigenvalues' and (not in_kpoints_opt):
                    self.eigenvalues = self._parse_eigen(elem)
                elif parse_projected_eigen and tag == 'projected' and (not in_kpoints_opt):
                    self.projected_eigenvalues, self.projected_magnetisation = self._parse_projected_eigen(elem)
                elif tag in ('eigenvalues_kpoints_opt', 'projected_kpoints_opt'):
                    if self.kpoints_opt_props is None:
                        self.kpoints_opt_props = KpointOptProps()
                    in_kpoints_opt = False
                    self.kpoints_opt_props.eigenvalues = self._parse_eigen(elem.find('eigenvalues'))
                    if tag == 'eigenvalues_kpoints_opt':
                        self.kpoints_opt_props.kpoints, self.kpoints_opt_props.actual_kpoints, self.kpoints_opt_props.actual_kpoints_weights = self._parse_kpoints(elem.find('kpoints'))
                    elif parse_projected_eigen:
                        self.kpoints_opt_props.projected_eigenvalues, self.kpoints_opt_props.projected_magnetisation = self._parse_projected_eigen(elem)
                elif tag == 'structure' and elem.attrib.get('name') == 'finalpos':
                    self.final_structure = self._parse_structure(elem)
        self.vasp_version = self.generator['version']
        if parse_potcar_file:
            self.update_potcar_spec(parse_potcar_file)

    def as_dict(self):
        """JSON-serializable dict representation."""
        dct = {'vasp_version': self.vasp_version, 'has_vasp_completed': True, 'nsites': len(self.final_structure)}
        comp = self.final_structure.composition
        dct['unit_cell_formula'] = comp.as_dict()
        dct['reduced_cell_formula'] = Composition(comp.reduced_formula).as_dict()
        dct['pretty_formula'] = comp.reduced_formula
        dct['is_hubbard'] = self.is_hubbard
        dct['hubbards'] = self.hubbards
        unique_symbols = sorted(set(self.atomic_symbols))
        dct['elements'] = unique_symbols
        dct['nelements'] = len(unique_symbols)
        dct['run_type'] = self.run_type
        vin = {'incar': dict(self.incar), 'crystal': self.final_structure.as_dict(), 'kpoints': self.kpoints.as_dict()}
        actual_kpts = [{'abc': list(self.actual_kpoints[i]), 'weight': self.actual_kpoints_weights[i]} for i in range(len(self.actual_kpoints))]
        vin['kpoints']['actual_points'] = actual_kpts
        if (kpt_opt_props := getattr(self, 'kpoints_opt_props', None)):
            vin['kpoints_opt'] = kpt_opt_props.kpoints.as_dict()
            actual_kpts = [{'abc': list(kpt_opt_props.actual_kpoints[idx]), 'weight': kpt_opt_props.actual_kpoints_weights[idx]} for idx in range(len(kpt_opt_props.actual_kpoints))]
            vin['kpoints_opt']['actual_kpoints'] = actual_kpts
            vin['nkpoints_opt'] = len(actual_kpts)
        vin['potcar'] = [s.split(' ')[1] for s in self.potcar_symbols]
        vin['potcar_spec'] = self.potcar_spec
        vin['potcar_type'] = [s.split(' ')[0] for s in self.potcar_symbols]
        vin['parameters'] = dict(self.parameters)
        vin['lattice_rec'] = self.final_structure.lattice.reciprocal_lattice.as_dict()
        dct['input'] = vin
        vout = {'crystal': self.final_structure.as_dict(), 'efermi': self.efermi}
        if self.eigenvalues:
            eigen = defaultdict(dict)
            for spin, values in self.eigenvalues.items():
                for idx, val in enumerate(values):
                    eigen[idx][str(spin)] = val
            vout['eigenvalues'] = eigen
            gap, cbm, vbm, is_direct = self.eigenvalue_band_properties
            vout.update({'bandgap': gap, 'cbm': cbm, 'vbm': vbm, 'is_gap_direct': is_direct})
            if self.projected_eigenvalues:
                peigen = [{} for _ in eigen]
                for spin, val in self.projected_eigenvalues.items():
                    for kpoint_index, vv in enumerate(val):
                        if str(spin) not in peigen[kpoint_index]:
                            peigen[kpoint_index][str(spin)] = vv
                vout['projected_eigenvalues'] = peigen
        if kpt_opt_props and kpt_opt_props.eigenvalues:
            eigen = {str(spin): v.tolist() for spin, v in kpt_opt_props.eigenvalues.items()}
            vout['eigenvalues_kpoints_opt'] = eigen
            if kpt_opt_props.projected_eigenvalues:
                vout['projected_eigenvalues_kpoints_opt'] = {str(spin): v.tolist() for spin, v in kpt_opt_props.projected_eigenvalues.items()}
        dct['output'] = vout
        return jsanitize(dct, strict=True)