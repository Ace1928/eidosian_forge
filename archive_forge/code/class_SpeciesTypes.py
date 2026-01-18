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
class SpeciesTypes(SimpleVaspHeaderParser):
    """Parse species types.

    Example line:
    " POTCAR:    PAW_PBE Ni 02Aug2007"

    We must parse this multiple times, as it's scattered in the header.
    So this class has to simply parse the entire header.
    """
    LINE_DELIMITER = 'POTCAR:'

    def __init__(self, *args, **kwargs):
        self._species = []
        super().__init__(*args, **kwargs)

    @property
    def species(self) -> List[str]:
        """Internal storage of each found line.
        Will contain the double counting.
        Use the get_species() method to get the un-doubled list."""
        return self._species

    def get_species(self) -> List[str]:
        """The OUTCAR will contain two 'POTCAR:' entries per species.
        This method only returns the first half,
        effectively removing the double counting.
        """
        idx = sum(divmod(len(self.species), 2))
        return list(self.species[:idx])

    def _make_returnval(self) -> _RESULT:
        """Construct the return value for the "parse" method"""
        return {'species': self.get_species()}

    def parse(self, cursor: _CURSOR, lines: _CHUNK) -> _RESULT:
        line = lines[cursor].strip()
        parts = line.split()
        if '1/r potential' in line:
            idx = 1
        else:
            idx = 2
        sym = parts[idx]
        sym = sym.split('_')[0]
        sym = ''.join([s for s in sym if s.isalpha()])
        if sym not in atomic_numbers:
            raise ParseError(f'Found an unexpected symbol {sym} in line {line}')
        self.species.append(sym)
        return self._make_returnval()