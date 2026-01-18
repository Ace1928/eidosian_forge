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
class Magmom(VaspChunkPropertyParser):

    def has_property(self, cursor: _CURSOR, lines: _CHUNK) -> bool:
        """ We need to check for two separate delimiter strings,
        to ensure we are at the right place """
        line = lines[cursor]
        if 'number of electron' in line:
            parts = line.split()
            if len(parts) > 5 and parts[0].strip() != 'NELECT':
                return True
        return False

    def parse(self, cursor: _CURSOR, lines: _CHUNK) -> _RESULT:
        line = self.get_line(cursor, lines)
        parts = line.split()
        idx = parts.index('magnetization') + 1
        magmom_lst = parts[idx:]
        if len(magmom_lst) != 1:
            warn('Non-collinear spin is not yet implemented. Setting magmom to x value.')
        magmom = float(magmom_lst[0])
        return {'magmom': magmom}