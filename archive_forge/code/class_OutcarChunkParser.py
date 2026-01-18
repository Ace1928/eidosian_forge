from abc import ABC, abstractmethod
from typing import (Dict, Any, Sequence, TextIO, Iterator, Optional, Union,
import re
from warnings import warn
from pathlib import Path, PurePath
import numpy as np
import ase
from ase import Atoms
from ase.data import atomic_numbers
from ase.io import ParseError, read
from ase.io.utils import ImageChunk
from ase.calculators.singlepoint import SinglePointDFTCalculator, SinglePointKPoint
class OutcarChunkParser(ChunkParser):
    """Class for parsing a chunk of an OUTCAR."""

    def __init__(self, header: _HEADER=None, parsers: Sequence[VaspChunkPropertyParser]=None):
        global default_chunk_parsers
        parsers = parsers or default_chunk_parsers.make_parsers()
        super().__init__(parsers, header=header)

    def build(self, lines: _CHUNK) -> Atoms:
        """Apply outcar chunk parsers, and build an atoms object"""
        self.update_parser_headers()
        results = self.parse(lines)
        symbols = self.header['symbols']
        constraint = self.header.get('constraint', None)
        atoms_kwargs = dict(symbols=symbols, constraint=constraint, pbc=True)
        for prop in ('positions', 'cell'):
            try:
                atoms_kwargs[prop] = results.pop(prop)
            except KeyError:
                raise ParseError('Did not find required property {} during parse.'.format(prop))
        atoms = Atoms(**atoms_kwargs)
        kpts = results.pop('kpts', None)
        calc = SinglePointDFTCalculator(atoms, **results)
        if kpts is not None:
            calc.kpts = kpts
        calc.name = 'vasp'
        atoms.calc = calc
        return atoms