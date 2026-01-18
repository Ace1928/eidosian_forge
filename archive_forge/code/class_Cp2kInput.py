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
class Cp2kInput(Section):
    """
    Special instance of 'Section' class that is meant to represent the overall cp2k input.
    Distinguishes itself from Section by overriding get_str() to not print this section's
    title and by implementing the file i/o.
    """

    def __init__(self, name: str='CP2K_INPUT', subsections: dict | None=None, **kwargs):
        """Initialize Cp2kInput by calling the super."""
        self.name = name
        self.subsections = subsections or {}
        self.kwargs = kwargs
        description = 'CP2K Input'
        super().__init__(name, repeats=False, description=description, section_parameters=[], subsections=subsections, **kwargs)

    def get_str(self):
        """Get string representation of the Cp2kInput."""
        string = ''
        for v in self.subsections.values():
            string += v.get_str()
        return string

    @classmethod
    def _from_dict(cls, dct):
        """Initialize from a dictionary."""
        return Cp2kInput('CP2K_INPUT', subsections=getattr(__import__(dct['@module'], globals(), locals(), dct['@class'], 0), dct['@class']).from_dict(dct).subsections)

    @classmethod
    def from_file(cls, filename: str | Path) -> Self:
        """Initialize from a file."""
        with zopen(filename, mode='rt') as file:
            txt = preprocessor(file.read(), os.path.dirname(file.name))
            return cls.from_str(txt)

    @classmethod
    def from_str(cls, s: str) -> Self:
        """Initialize from a string."""
        lines = s.splitlines()
        lines = [line.replace('\t', '') for line in lines]
        lines = [line.strip() for line in lines]
        lines = [line for line in lines if line]
        return cls.from_lines(lines)

    @classmethod
    def from_lines(cls, lines: list | tuple) -> Self:
        """Helper method to read lines of file."""
        cp2k_input = Cp2kInput('CP2K_INPUT', subsections={})
        Cp2kInput._from_lines(cp2k_input, lines)
        return cp2k_input

    def _from_lines(self, lines):
        """Helper method, reads lines of text to get a Cp2kInput."""
        current = self.name
        description = ''
        for line in lines:
            if line.startswith(('!', '#')):
                description += line[1:].strip()
            elif line.upper().startswith('&END'):
                current = '/'.join(current.split('/')[:-1])
            elif line.startswith('&'):
                name, subsection_params = (line.split()[0][1:], line.split()[1:])
                subsection_params = [] if len(subsection_params) == 1 and subsection_params[0].upper() in ('T', 'TRUE', 'F', 'FALSE', 'ON') else subsection_params
                alias = f'{name} {' '.join(subsection_params)}' if subsection_params else None
                sec = Section(name, section_parameters=subsection_params, alias=alias, subsections={}, description=description)
                description = ''
                if (tmp := self.by_path(current).get_section(sec.alias or sec.name)):
                    if isinstance(tmp, SectionList):
                        self.by_path(current)[sec.alias or sec.name].append(sec)
                    else:
                        self.by_path(current)[sec.alias or sec.name] = SectionList(sections=[tmp, sec])
                else:
                    self.by_path(current).insert(sec)
                current = f'{current}/{alias or name}'
            else:
                kwd = Keyword.from_str(line)
                if (tmp := self.by_path(current).get(kwd.name)):
                    if isinstance(tmp, KeywordList):
                        self.by_path(current).get(kwd.name).append(kwd)
                    elif isinstance(self.by_path(current), SectionList):
                        self.by_path(current)[-1][kwd.name] = KeywordList(keywords=[tmp, kwd])
                    else:
                        self.by_path(current)[kwd.name] = KeywordList(keywords=[kwd, tmp])
                elif isinstance(self.by_path(current), SectionList):
                    self.by_path(current)[-1].keywords[kwd.name] = kwd
                else:
                    self.by_path(current).keywords[kwd.name] = kwd

    def write_file(self, input_filename: str='cp2k.inp', output_dir: str='.', make_dir_if_not_present: bool=True):
        """Write input to a file.

        Args:
            input_filename (str, optional): Defaults to "cp2k.inp".
            output_dir (str, optional): Defaults to ".".
            make_dir_if_not_present (bool, optional): Defaults to True.
        """
        if not os.path.isdir(output_dir) and make_dir_if_not_present:
            os.mkdir(output_dir)
        filepath = os.path.join(output_dir, input_filename)
        with open(filepath, mode='w', encoding='utf-8') as file:
            file.write(self.get_str())