from __future__ import annotations
import abc
import collections
import hashlib
import logging
import os
import shutil
import sys
import tempfile
import traceback
from collections import defaultdict, namedtuple
from typing import TYPE_CHECKING
from xml.etree import ElementTree as Et
import numpy as np
from monty.collections import AttrDict, Namespace
from monty.functools import lazy_property
from monty.itertools import iterator_from_slice
from monty.json import MontyDecoder, MSONable
from monty.os.path import find_exts
from tabulate import tabulate
from pymatgen.core import Element
from pymatgen.core.xcfunc import XcFunc
from pymatgen.io.core import ParseError
from pymatgen.util.plotting import add_fig_kwargs, get_ax_fig
class PseudoParser:
    """
    Responsible for parsing pseudopotential files and returning pseudopotential objects.

    Usage:
        pseudo = PseudoParser().parse("filename")
    """
    Error = PseudoParseError
    ppdesc = namedtuple('ppdesc', 'pspcod name psp_type format')
    _PSPCODES = {1: ppdesc(1, 'TM', 'NC', None), 2: ppdesc(2, 'GTH', 'NC', None), 3: ppdesc(3, 'HGH', 'NC', None), 4: ppdesc(4, 'Teter', 'NC', None), 6: ppdesc(6, 'FHI', 'NC', None), 7: ppdesc(6, 'PAW_abinit_text', 'PAW', None), 8: ppdesc(8, 'ONCVPSP', 'NC', None), 10: ppdesc(10, 'HGHK', 'NC', None)}
    del ppdesc

    def __init__(self):
        self._parsed_paths = []
        self._wrong_paths = []

    def scan_directory(self, dirname, exclude_exts=(), exclude_fnames=()):
        """
        Analyze the files contained in directory dirname.

        Args:
            dirname: directory path
            exclude_exts: list of file extensions that should be skipped.
            exclude_fnames: list of file names that should be skipped.

        Returns:
            List of pseudopotential objects.
        """
        for i, ext in enumerate(exclude_exts):
            if not ext.strip().startswith('.'):
                exclude_exts[i] = '.' + ext.strip()
        paths = []
        for fname in os.listdir(dirname):
            _root, ext = os.path.splitext(fname)
            path = os.path.join(dirname, fname)
            if ext in exclude_exts or fname in exclude_fnames or fname.startswith('.') or (not os.path.isfile(path)):
                continue
            paths.append(path)
        pseudos = []
        for path in paths:
            try:
                pseudo = self.parse(path)
            except Exception:
                pseudo = None
            if pseudo is not None:
                pseudos.append(pseudo)
                self._parsed_paths.extend(path)
            else:
                self._wrong_paths.extend(path)
        return pseudos

    def read_ppdesc(self, filename):
        """
        Read the pseudopotential descriptor from filename.

        Returns:
            Pseudopotential descriptor. None if filename is not a valid pseudopotential file.

        Raises:
            `PseudoParseError` if fileformat is not supported.
        """
        if filename.endswith('.xml'):
            raise self.Error('XML pseudo not supported yet')
        lines = _read_nlines(filename, 80)
        for lineno, line in enumerate(lines, start=1):
            if lineno == 3:
                try:
                    tokens = line.split()
                    pspcod, _pspxc = map(int, tokens[:2])
                except Exception:
                    msg = f'{filename}: Cannot parse pspcod, pspxc in line\n {line}'
                    logger.critical(msg)
                    return None
                if pspcod not in self._PSPCODES:
                    raise self.Error(f"{filename}: Don't know how to handle pspcod={pspcod!r}\n")
                ppdesc = self._PSPCODES[pspcod]
                if pspcod == 7:
                    tokens = lines[lineno].split()
                    pspfmt, _creatorID = tokens[:2]
                    ppdesc = ppdesc._replace(format=pspfmt)
                return ppdesc
        return None

    def parse(self, filename):
        """
        Read and parse a pseudopotential file. Main entry point for client code.

        Returns:
            pseudopotential object or None if filename is not a valid pseudopotential file.
        """
        path = os.path.abspath(filename)
        if filename.endswith('.xml'):
            return PawXmlSetup(path)
        ppdesc = self.read_ppdesc(path)
        if ppdesc is None:
            logger.critical(f'Cannot find ppdesc in {path}')
            return None
        psp_type = ppdesc.psp_type
        parsers = {'FHI': NcAbinitHeader.fhi_header, 'GTH': NcAbinitHeader.gth_header, 'TM': NcAbinitHeader.tm_header, 'Teter': NcAbinitHeader.tm_header, 'HGH': NcAbinitHeader.hgh_header, 'HGHK': NcAbinitHeader.hgh_header, 'ONCVPSP': NcAbinitHeader.oncvpsp_header, 'PAW_abinit_text': PawAbinitHeader.paw_header}
        try:
            header = parsers[ppdesc.name](path, ppdesc)
        except Exception:
            raise self.Error(f'{path}:\n{straceback()}')
        if psp_type == 'NC':
            pseudo = NcAbinitPseudo(path, header)
        elif psp_type == 'PAW':
            pseudo = PawAbinitPseudo(path, header)
        else:
            raise NotImplementedError('psp_type not in [NC, PAW]')
        return pseudo