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
class EFermi(SimpleVaspChunkParser):
    LINE_DELIMITER = 'E-fermi :'

    def parse(self, cursor: _CURSOR, lines: _CHUNK) -> _RESULT:
        line = self.get_line(cursor, lines)
        parts = line.split()
        efermi = float(parts[2])
        return {'efermi': efermi}