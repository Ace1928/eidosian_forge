from __future__ import annotations
import codecs
import contextlib
import hashlib
import itertools
import json
import logging
import math
import os
import re
import subprocess
import warnings
from collections import namedtuple
from enum import Enum, unique
from glob import glob
from hashlib import sha256
from typing import TYPE_CHECKING, Any, Literal, cast
import numpy as np
import scipy.constants as const
from monty.io import zopen
from monty.json import MontyDecoder, MSONable
from monty.os import cd
from monty.os.path import zpath
from monty.serialization import dumpfn, loadfn
from tabulate import tabulate
from pymatgen.core import SETTINGS, Element, Lattice, Structure, get_el_sp
from pymatgen.electronic_structure.core import Magmom
from pymatgen.util.io_utils import clean_lines
from pymatgen.util.string import str_delimited
class PotcarSingle:
    """
    Object for a **single** POTCAR. The builder assumes the POTCAR contains
    the complete untouched data in "data" as a string and a dict of keywords.

    Attributes:
        data (str): POTCAR data as a string.
        keywords (dict): Keywords parsed from the POTCAR as a dict. All keywords are also
            accessible as attributes in themselves. E.g., potcar.enmax, potcar.encut, etc.

    md5 hashes of the entire POTCAR file and the actual data are validated
    against a database of known good hashes. Appropriate warnings or errors
    are raised if a POTCAR hash fails validation.
    """
    functional_dir = dict(PBE='POT_GGA_PAW_PBE', PBE_52='POT_GGA_PAW_PBE_52', PBE_52_W_HASH='POTPAW_PBE_52', PBE_54='POT_GGA_PAW_PBE_54', PBE_54_W_HASH='POTPAW_PBE_54', PBE_64='POT_PAW_PBE_64', LDA='POT_LDA_PAW', LDA_52='POT_LDA_PAW_52', LDA_52_W_HASH='POTPAW_LDA_52', LDA_54='POT_LDA_PAW_54', LDA_54_W_HASH='POTPAW_LDA_54', LDA_64='POT_LDA_PAW_64', PW91='POT_GGA_PAW_PW91', LDA_US='POT_LDA_US', PW91_US='POT_GGA_US_PW91', Perdew_Zunger81='POT_LDA_PAW')
    functional_tags = {'pe': {'name': 'PBE', 'class': 'GGA'}, '91': {'name': 'PW91', 'class': 'GGA'}, 'rp': {'name': 'revPBE', 'class': 'GGA'}, 'am': {'name': 'AM05', 'class': 'GGA'}, 'ps': {'name': 'PBEsol', 'class': 'GGA'}, 'pw': {'name': 'PW86', 'class': 'GGA'}, 'lm': {'name': 'Langreth-Mehl-Hu', 'class': 'GGA'}, 'pb': {'name': 'Perdew-Becke', 'class': 'GGA'}, 'ca': {'name': 'Perdew-Zunger81', 'class': 'LDA'}, 'hl': {'name': 'Hedin-Lundquist', 'class': 'LDA'}, 'wi': {'name': 'Wigner Interpolation', 'class': 'LDA'}}
    parse_functions = dict(LULTRA=_parse_bool, LUNSCR=_parse_bool, LCOR=_parse_bool, LPAW=_parse_bool, EATOM=_parse_float, RPACOR=_parse_float, POMASS=_parse_float, ZVAL=_parse_float, RCORE=_parse_float, RWIGS=_parse_float, ENMAX=_parse_float, ENMIN=_parse_float, EMMIN=_parse_float, EAUG=_parse_float, DEXC=_parse_float, RMAX=_parse_float, RAUG=_parse_float, RDEP=_parse_float, RDEPT=_parse_float, QCUT=_parse_float, QGAM=_parse_float, RCLOC=_parse_float, IUNSCR=_parse_int, ICORE=_parse_int, NDATA=_parse_int, VRHFIN=str.strip, LEXCH=str.strip, TITEL=str.strip, STEP=_parse_list, RRKJ=_parse_list, GGA=_parse_list, SHA256=str.strip, COPYR=str.strip)
    _potcar_summary_stats = loadfn(POTCAR_STATS_PATH)

    def __init__(self, data: str, symbol: str | None=None) -> None:
        """
        Args:
            data (str): Complete and single POTCAR file as a string.
            symbol (str): POTCAR symbol corresponding to the filename suffix e.g. "Tm_3" for POTCAR.TM_3".
                If not given, pymatgen will attempt to extract the symbol from the file itself. This is
                not always reliable!
        """
        self.data = data
        self.header = data.split('\n')[0].strip()
        match = re.search('(?s)(parameters from PSCTR are:.*?END of PSCTR-controll parameters)', data)
        search_lines = match.group(1) if match else ''
        keywords = {}
        for key, val in re.findall('(\\S+)\\s*=\\s*(.*?)(?=;|$)', search_lines, flags=re.MULTILINE):
            try:
                keywords[key] = self.parse_functions[key](val)
            except KeyError:
                warnings.warn(f'Ignoring unknown variable type {key}')
        PSCTR: dict[str, Any] = {}
        array_search = re.compile('(-*[0-9.]+)')
        orbitals = []
        descriptions = []
        atomic_config_match = re.search('(?s)Atomic configuration(.*?)Description', search_lines)
        if atomic_config_match:
            lines = atomic_config_match.group(1).splitlines()
            match = re.search('([0-9]+)', lines[1])
            num_entries = int(match.group(1)) if match else 0
            PSCTR['nentries'] = num_entries
            for line in lines[3:]:
                orbit = array_search.findall(line)
                if orbit:
                    orbitals.append(Orbital(int(orbit[0]), int(orbit[1]), float(orbit[2]), float(orbit[3]), float(orbit[4])))
            PSCTR['Orbitals'] = tuple(orbitals)
        description_string = re.search('(?s)Description\\s*\\n(.*?)Error from kinetic energy argument \\(eV\\)', search_lines)
        if description_string:
            for line in description_string.group(1).splitlines():
                description = array_search.findall(line)
                if description:
                    descriptions.append(OrbitalDescription(int(description[0]), float(description[1]), int(description[2]), float(description[3]), int(description[4]) if len(description) > 4 else None, float(description[5]) if len(description) > 4 else None))
        if descriptions:
            PSCTR['OrbitalDescriptions'] = tuple(descriptions)
        rrkj_kinetic_energy_string = re.search('(?s)Error from kinetic energy argument \\(eV\\)\\s*\\n(.*?)END of PSCTR-controll parameters', search_lines)
        rrkj_array = []
        if rrkj_kinetic_energy_string:
            for line in rrkj_kinetic_energy_string.group(1).splitlines():
                if '=' not in line:
                    rrkj_array += _parse_list(line.strip('\n'))
            if rrkj_array:
                PSCTR['RRKJ'] = tuple(rrkj_array)
        self.keywords = dict(sorted({**PSCTR, **keywords}.items()))
        if symbol:
            self._symbol = symbol
        else:
            try:
                self._symbol = keywords['TITEL'].split(' ')[1].strip()
            except IndexError:
                self._symbol = keywords['TITEL'].strip()
        if not self.is_valid:
            warnings.warn(f"POTCAR data with symbol {self.symbol} is not known to pymatgen. Your POTCAR may be corrupted or pymatgen's POTCAR database is incomplete.", UnknownPotcarWarning)

    def __str__(self) -> str:
        return f'{self.data}\n'

    @property
    def electron_configuration(self) -> list[tuple[int, str, int]] | None:
        """Electronic configuration of the PotcarSingle."""
        if not self.nelectrons.is_integer():
            warnings.warn('POTCAR has non-integer charge, electron configuration not well-defined.')
            return None
        el = Element.from_Z(self.atomic_no)
        full_config = el.full_electronic_structure
        nelect = self.nelectrons
        config = []
        while nelect > 0:
            e = full_config.pop(-1)
            config.append(e)
            nelect -= e[-1]
        return config

    def write_file(self, filename: str) -> None:
        """Write PotcarSingle to a file.

        Args:
            filename (str): Filename to write to.
        """
        with zopen(filename, mode='wt') as file:
            file.write(str(self))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PotcarSingle):
            return NotImplemented
        return self.data == other.data and self.keywords == other.keywords

    def copy(self) -> PotcarSingle:
        """Returns a copy of the PotcarSingle.

        Returns:
            PotcarSingle
        """
        return PotcarSingle(self.data, symbol=self.symbol)

    @classmethod
    def from_file(cls, filename: str) -> Self:
        """Reads PotcarSingle from file.

        Args:
            filename: Filename.

        Returns:
            PotcarSingle
        """
        match = re.search('(?<=POTCAR\\.)(.*)(?=.gz)', str(filename))
        symbol = match[0] if match else ''
        try:
            with zopen(filename, mode='rt') as file:
                return cls(file.read(), symbol=symbol or None)
        except UnicodeDecodeError:
            warnings.warn('POTCAR contains invalid unicode errors. We will attempt to read it by ignoring errors.')
            with codecs.open(filename, 'r', encoding='utf-8', errors='ignore') as file:
                return cls(file.read(), symbol=symbol or None)

    @classmethod
    def from_symbol_and_functional(cls, symbol: str, functional: str | None=None) -> Self:
        """Makes a PotcarSingle from a symbol and functional.

        Args:
            symbol (str): Symbol, e.g., Li_sv
            functional (str): Functional, e.g., PBE

        Returns:
            PotcarSingle
        """
        functional = functional or SETTINGS.get('PMG_DEFAULT_FUNCTIONAL', 'PBE')
        assert isinstance(functional, str)
        funcdir = cls.functional_dir[functional]
        PMG_VASP_PSP_DIR = SETTINGS.get('PMG_VASP_PSP_DIR')
        if PMG_VASP_PSP_DIR is None:
            raise ValueError(f'No POTCAR for {symbol} with functional={functional!r} found. Please set the PMG_VASP_PSP_DIR in .pmgrc.yaml.')
        paths_to_try = [os.path.join(PMG_VASP_PSP_DIR, funcdir, f'POTCAR.{symbol}'), os.path.join(PMG_VASP_PSP_DIR, funcdir, symbol, 'POTCAR')]
        for path in paths_to_try:
            path = os.path.expanduser(path)
            path = zpath(path)
            if os.path.isfile(path):
                return cls.from_file(path)
        raise OSError(f'You do not have the right POTCAR with functional={functional!r} and symbol={symbol!r} in your PMG_VASP_PSP_DIR={PMG_VASP_PSP_DIR!r}. Paths tried: {paths_to_try}')

    @property
    def element(self) -> str:
        """Attempt to return the atomic symbol based on the VRHFIN keyword."""
        element = self.keywords['VRHFIN'].split(':')[0].strip()
        try:
            return Element(element).symbol
        except ValueError:
            if element == 'X':
                return 'Xe'
            return Element(self.symbol.split('_')[0]).symbol

    @property
    def atomic_no(self) -> int:
        """Attempt to return the atomic number based on the VRHFIN keyword."""
        return Element(self.element).Z

    @property
    def nelectrons(self) -> float:
        """Number of electrons"""
        return self.zval

    @property
    def symbol(self) -> str:
        """The POTCAR symbol, e.g. W_pv"""
        return self._symbol

    @property
    def potential_type(self) -> Literal['NC', 'PAW', 'US']:
        """Type of PSP. E.g., US, PAW, etc."""
        if self.lultra:
            return 'US'
        if self.lpaw:
            return 'PAW'
        return 'NC'

    @property
    def functional(self) -> str | None:
        """Functional associated with PotcarSingle."""
        return self.functional_tags.get(self.LEXCH.lower(), {}).get('name')

    @property
    def functional_class(self):
        """Functional class associated with PotcarSingle."""
        return self.functional_tags.get(self.LEXCH.lower(), {}).get('class')

    def verify_potcar(self) -> tuple[bool, bool]:
        """
        Attempts to verify the integrity of the POTCAR data.

        This method checks the whole file (removing only the SHA256
        metadata) against the SHA256 hash in the header if this is found.
        If no SHA256 hash is found in the file, the file hash (md5 hash of the
        whole file) is checked against all POTCAR file hashes known to pymatgen.

        Returns:
            tuple[bool, bool]: has_sha256 and passed_hash_check are returned.
        """
        if self.hash_sha256_from_file:
            has_sha256 = True
            hash_is_valid = self.hash_sha256_from_file == self.sha256_computed_file_hash
        else:
            has_sha256 = False
            md5_file_hash = self.md5_computed_file_hash
            hash_is_valid = md5_file_hash in VASP_POTCAR_HASHES
        return (has_sha256, hash_is_valid)

    def identify_potcar(self, mode: Literal['data', 'file']='data', data_tol: float=1e-06) -> tuple[list[str], list[str]]:
        """
        Identify the symbol and compatible functionals associated with this PotcarSingle.

        This method checks the summary statistics of either the POTCAR metadadata
        (PotcarSingle._summary_stats[key]["header"] for key in ("keywords", "stats") )
        or the entire POTCAR file (PotcarSingle._summary_stats) against a database
        of hashes for POTCARs distributed with VASP 5.4.4.

        Args:
            mode ('data' | 'file'): 'data' mode checks the POTCAR header keywords and stats only
                while 'file' mode checks the entire summary stats.
            data_tol (float): Tolerance for comparing the summary statistics of the POTCAR
                with the reference statistics.

        Returns:
            symbol (list): List of symbols associated with the PotcarSingle
            potcar_functionals (list): List of potcar functionals associated with
                the PotcarSingle
        """
        if mode == 'data':
            check_modes = ['header']
        elif mode == 'file':
            check_modes = ['header', 'data']
        else:
            raise ValueError(f"Bad mode={mode!r}. Choose 'data' or 'file'.")
        identity: dict[str, list] = {'potcar_functionals': [], 'potcar_symbols': []}
        for func in self.functional_dir:
            for ref_psp in self._potcar_summary_stats[func].get(self.TITEL.replace(' ', ''), []):
                if self.VRHFIN.replace(' ', '') != ref_psp['VRHFIN']:
                    continue
                key_match = all((set(ref_psp['keywords'][key]) == set(self._summary_stats['keywords'][key]) for key in check_modes))
                data_diff = [abs(ref_psp['stats'][key][stat] - self._summary_stats['stats'][key][stat]) for stat in ['MEAN', 'ABSMEAN', 'VAR', 'MIN', 'MAX'] for key in check_modes]
                data_match = all(np.array(data_diff) < data_tol)
                if key_match and data_match:
                    identity['potcar_functionals'].append(func)
                    identity['potcar_symbols'].append(ref_psp['symbol'])
        for key, values in identity.items():
            if len(values) == 0:
                return ([], [])
            identity[key] = list(set(values))
        return (identity['potcar_functionals'], identity['potcar_symbols'])

    def identify_potcar_hash_based(self, mode: Literal['data', 'file']='data'):
        """
        Identify the symbol and compatible functionals associated with this PotcarSingle.

        This method checks the md5 hash of either the POTCAR metadadata (PotcarSingle.md5_header_hash)
        or the entire POTCAR file (PotcarSingle.md5_computed_file_hash) against a database
        of hashes for POTCARs distributed with VASP 5.4.4.

        Args:
            mode ('data' | 'file'): 'data' mode checks the hash of the POTCAR metadata in self.keywords,
                while 'file' mode checks the hash of the entire POTCAR file.

        Returns:
            symbol (list): List of symbols associated with the PotcarSingle
            potcar_functionals (list): List of potcar functionals associated with
                the PotcarSingle
        """
        mapping_dict = {'potUSPP_GGA': {'pymatgen_key': 'PW91_US', 'vasp_description': 'Ultrasoft pseudo potentialsfor LDA and PW91 (dated 2002-08-20 and 2002-04-08,respectively). These files are outdated, notsupported and only distributed as is.'}, 'potUSPP_LDA': {'pymatgen_key': 'LDA_US', 'vasp_description': 'Ultrasoft pseudo potentialsfor LDA and PW91 (dated 2002-08-20 and 2002-04-08,respectively). These files are outdated, notsupported and only distributed as is.'}, 'potpaw_GGA': {'pymatgen_key': 'PW91', 'vasp_description': 'The LDA, PW91 and PBE PAW datasets(snapshot: 05-05-2010, 19-09-2006 and 06-05-2010,respectively). These files are outdated, notsupported and only distributed as is.'}, 'potpaw_LDA': {'pymatgen_key': 'Perdew-Zunger81', 'vasp_description': 'The LDA, PW91 and PBE PAW datasets(snapshot: 05-05-2010, 19-09-2006 and 06-05-2010,respectively). These files are outdated, notsupported and only distributed as is.'}, 'potpaw_LDA.52': {'pymatgen_key': 'LDA_52', 'vasp_description': "LDA PAW datasets version 52,including the early GW variety (snapshot 19-04-2012).When read by VASP these files yield identical resultsas the files distributed in 2012 ('unvie' release)."}, 'potpaw_LDA.54': {'pymatgen_key': 'LDA_54', 'vasp_description': 'LDA PAW datasets version 54,including the GW variety (original release 2015-09-04).When read by VASP these files yield identical results asthe files distributed before.'}, 'potpaw_PBE': {'pymatgen_key': 'PBE', 'vasp_description': 'The LDA, PW91 and PBE PAW datasets(snapshot: 05-05-2010, 19-09-2006 and 06-05-2010,respectively). These files are outdated, notsupported and only distributed as is.'}, 'potpaw_PBE.52': {'pymatgen_key': 'PBE_52', 'vasp_description': 'PBE PAW datasets version 52,including early GW variety (snapshot 19-04-2012).When read by VASP these files yield identicalresults as the files distributed in 2012.'}, 'potpaw_PBE.54': {'pymatgen_key': 'PBE_54', 'vasp_description': 'PBE PAW datasets version 54,including the GW variety (original release 2015-09-04).When read by VASP these files yield identical results asthe files distributed before.'}, 'unvie_potpaw.52': {'pymatgen_key': 'unvie_LDA_52', 'vasp_description': 'files released previouslyfor vasp.5.2 (2012-04) and vasp.5.4 (2015-09-04) by univie.'}, 'unvie_potpaw.54': {'pymatgen_key': 'unvie_LDA_54', 'vasp_description': 'files released previouslyfor vasp.5.2 (2012-04) and vasp.5.4 (2015-09-04) by univie.'}, 'unvie_potpaw_PBE.52': {'pymatgen_key': 'unvie_PBE_52', 'vasp_description': 'files released previouslyfor vasp.5.2 (2012-04) and vasp.5.4 (2015-09-04) by univie.'}, 'unvie_potpaw_PBE.54': {'pymatgen_key': 'unvie_PBE_52', 'vasp_description': 'files released previouslyfor vasp.5.2 (2012-04) and vasp.5.4 (2015-09-04) by univie.'}}
        if mode == 'data':
            hash_db = PYMATGEN_POTCAR_HASHES
            potcar_hash = self.md5_header_hash
        elif mode == 'file':
            hash_db = VASP_POTCAR_HASHES
            potcar_hash = self.md5_computed_file_hash
        else:
            raise ValueError(f"Bad mode={mode!r}. Choose 'data' or 'file'.")
        if (identity := hash_db.get(potcar_hash)):
            potcar_functionals = [*{mapping_dict[i]['pymatgen_key'] for i in identity['potcar_functionals']}]
            return (potcar_functionals, identity['potcar_symbols'])
        return ([], [])

    @property
    def hash_sha256_from_file(self) -> str | None:
        """SHA256 hash of the POTCAR file as read from the file. None if no SHA256 hash is found."""
        if (sha256 := getattr(self, 'SHA256', None)):
            return sha256.split()[0]
        return None

    @property
    def sha256_computed_file_hash(self) -> str:
        """Computes a SHA256 hash of the PotcarSingle EXCLUDING lines starting with 'SHA256' and 'COPYR'."""
        potcar_list = self.data.split('\n')
        potcar_to_hash = [line for line in potcar_list if not line.strip().startswith(('SHA256', 'COPYR'))]
        potcar_to_hash_str = '\n'.join(potcar_to_hash)
        return sha256(potcar_to_hash_str.encode('utf-8')).hexdigest()

    @property
    def md5_computed_file_hash(self) -> str:
        """md5 hash of the entire PotcarSingle."""
        md5 = hashlib.new('md5', usedforsecurity=False)
        md5.update(self.data.encode('utf-8'))
        return md5.hexdigest()

    @property
    def md5_header_hash(self) -> str:
        """Computes a md5 hash of the metadata defining the PotcarSingle."""
        hash_str = ''
        for k, v in self.keywords.items():
            if k in ('nentries', 'Orbitals', 'SHA256', 'COPYR'):
                continue
            hash_str += f'{k}'
            if isinstance(v, (bool, int)):
                hash_str += f'{v}'
            elif isinstance(v, float):
                hash_str += f'{v:.3f}'
            elif isinstance(v, (tuple, list)):
                for item in v:
                    if isinstance(item, float):
                        hash_str += f'{item:.3f}'
                    elif isinstance(item, (Orbital, OrbitalDescription)):
                        for item_v in item:
                            if isinstance(item_v, (int, str)):
                                hash_str += f'{item_v}'
                            elif isinstance(item_v, float):
                                hash_str += f'{item_v:.3f}'
                            else:
                                hash_str += f'{item_v}' if item_v else ''
            else:
                hash_str += v.replace(' ', '')
        self.hash_str = hash_str
        md5 = hashlib.new('md5', usedforsecurity=False)
        md5.update(hash_str.lower().encode('utf-8'))
        return md5.hexdigest()

    @property
    def is_valid(self) -> bool:
        """
        Check that POTCAR matches reference metadata.
        Parsed metadata is stored in self._summary_stats as a human-readable dict,
            self._summary_stats = {
                "keywords": {
                    "header": list[str],
                    "data": list[str],
                },
                "stats": {
                    "header": dict[float],
                    "data": dict[float],
                },
            }

        Rationale:
        Each POTCAR is structured as
            Header (self.keywords)
            Data (actual pseudopotential values in data blocks)

        For the Data block of POTCAR, there are unformatted data blocks
        of unknown length and contents/data type, e.g., you might see
            <float> <bool>
            <Data Keyword>
            <int> <int> <float>
            <float> ... <float>
            <Data Keyword>
            <float> ... <float>
        but this is impossible to process algorithmically without a full POTCAR schema.
        Note also that POTCARs can contain **different** data keywords

        All keywords found in the header, essentially self.keywords, and the data block
        (<Data Keyword> above) are stored in self._summary_stats["keywords"]

        To avoid issues of copyright, statistics (mean, mean of abs vals, variance, max, min)
        for the numeric values in the header and data sections of POTCAR are stored
        in self._summary_stats["stats"]

        tol is then used to match statistical values within a tolerance
        """
        possible_potcar_matches = []
        for func in self.functional_dir:
            for titel_no_spc in self._potcar_summary_stats[func]:
                if self.TITEL.replace(' ', '') == titel_no_spc:
                    for potcar_subvariant in self._potcar_summary_stats[func][titel_no_spc]:
                        if self.VRHFIN.replace(' ', '') == potcar_subvariant['VRHFIN']:
                            possible_potcar_matches.append({'POTCAR_FUNCTIONAL': func, 'TITEL': titel_no_spc, **potcar_subvariant})

        def parse_fortran_style_str(input_str: str) -> Any:
            """Parse any input string as bool, int, float, or failing that, str.
            Used to parse FORTRAN-generated POTCAR files where it's unknown
            a priori what type of data will be encountered.
            """
            input_str = input_str.strip()
            if input_str.lower() in {'t', 'f', 'true', 'false'}:
                return input_str[0].lower() == 't'
            if input_str.upper() == input_str.lower() and input_str[0].isnumeric():
                if '.' in input_str:
                    return float(input_str)
                return int(input_str)
            try:
                return float(input_str)
            except ValueError:
                return input_str
        psp_keys, psp_vals = ([], [])
        potcar_body = self.data.split('END of PSCTR-controll parameters\n')[1]
        for row in re.split('\\n+|;', potcar_body):
            tmp_str = ''
            for raw_val in row.split():
                parsed_val = parse_fortran_style_str(raw_val)
                if isinstance(parsed_val, str):
                    tmp_str += parsed_val.strip()
                elif isinstance(parsed_val, (float, int)):
                    psp_vals.append(parsed_val)
            if len(tmp_str) > 0:
                psp_keys.append(tmp_str.lower())
        keyword_vals = []
        for kwd in self.keywords:
            val = self.keywords[kwd]
            if isinstance(val, bool):
                keyword_vals.append(1.0 if val else 0.0)
            elif isinstance(val, (float, int)):
                keyword_vals.append(val)
            elif hasattr(val, '__len__'):
                keyword_vals += [num for num in val if isinstance(num, (float, int))]

        def data_stats(data_list: Sequence) -> dict:
            """Used for hash-less and therefore less brittle POTCAR validity checking."""
            arr = np.array(data_list)
            return {'MEAN': np.mean(arr), 'ABSMEAN': np.mean(np.abs(arr)), 'VAR': np.mean(arr ** 2), 'MIN': arr.min(), 'MAX': arr.max()}
        self._summary_stats = {'keywords': {'header': [kwd.lower() for kwd in self.keywords], 'data': psp_keys}, 'stats': {'header': data_stats(keyword_vals), 'data': data_stats(psp_vals)}}
        data_match_tol = 1e-06
        for ref_psp in possible_potcar_matches:
            key_match = all((set(ref_psp['keywords'][key]) == set(self._summary_stats['keywords'][key]) for key in ['header', 'data']))
            data_diff = [abs(ref_psp['stats'][key][stat] - self._summary_stats['stats'][key][stat]) for stat in ['MEAN', 'ABSMEAN', 'VAR', 'MIN', 'MAX'] for key in ['header', 'data']]
            data_match = all(np.array(data_diff) < data_match_tol)
            if key_match and data_match:
                return True
        return False

    def __getattr__(self, attr: str) -> Any:
        """Delegates attributes to keywords. For example, you can use potcarsingle.enmax to get the ENMAX of the POTCAR.

        For float type properties, they are converted to the correct float. By
        default, all energies in eV and all length scales are in Angstroms.
        """
        try:
            return self.keywords[attr.upper()]
        except Exception:
            raise AttributeError(attr)

    def __repr__(self) -> str:
        cls_name = type(self).__name__
        symbol, functional = (self.symbol, self.functional)
        TITEL, VRHFIN, n_valence_elec = (self.keywords.get(key) for key in ('TITEL', 'VRHFIN', 'ZVAL'))
        return f'{cls_name}(symbol={symbol!r}, functional={functional!r}, TITEL={TITEL!r}, VRHFIN={VRHFIN!r}, n_valence_elec={n_valence_elec:.0f})'