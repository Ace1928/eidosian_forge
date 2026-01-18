from __future__ import annotations
import abc
import copy
import hashlib
import itertools
import os
import re
import textwrap
import typing
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal
from monty.dev import deprecated
from monty.io import zopen
from monty.json import MSONable
from pymatgen.core import Element
from pymatgen.io.cp2k.utils import chunk, postprocessor, preprocessor
from pymatgen.io.vasp.inputs import Kpoints as VaspKpoints
from pymatgen.io.vasp.inputs import KpointsSupportedModes
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
class Dft(Section):
    """Controls the DFT parameters in Cp2k."""

    def __init__(self, basis_set_filenames: Iterable=('BASIS_MOLOPT',), potential_filename='GTH_POTENTIALS', uks: bool=True, wfn_restart_file_name: str | None=None, keywords: dict | None=None, subsections: dict | None=None, **kwargs):
        """Initialize the DFT section.

        Args:
            basis_set_filenames: Name of the file that contains the basis set
                information. Defaults to "BASIS_MOLOPT".
            potential_filename: Name of the file that contains the pseudopotential
                information. Defaults to "GTH_POTENTIALS".
            uks: Whether to run unrestricted Kohn Sham (spin polarized).
                Defaults to True.
            wfn_restart_file_name: Defaults to None.
            keywords: additional keywords to add.
            subsections: Any subsections to initialize with. Defaults to None.
        """
        self.basis_set_filenames = basis_set_filenames
        self.potential_filename = potential_filename
        self.uks = uks
        self.wfn_restart_file_name = wfn_restart_file_name
        keywords = keywords or {}
        subsections = subsections or {}
        description = 'Parameter needed by dft programs'
        _keywords = {'BASIS_SET_FILE_NAME': KeywordList([Keyword('BASIS_SET_FILE_NAME', k) for k in basis_set_filenames]), 'POTENTIAL_FILE_NAME': Keyword('POTENTIAL_FILE_NAME', potential_filename), 'UKS': Keyword('UKS', uks, description='Whether to run unrestricted Kohn Sham (i.e. spin polarized)')}
        if wfn_restart_file_name:
            _keywords['WFN_RESTART_FILE_NAME'] = Keyword('WFN_RESTART_FILE_NAME', wfn_restart_file_name)
        keywords.update(_keywords)
        super().__init__('DFT', description=description, keywords=keywords, subsections=subsections, **kwargs)