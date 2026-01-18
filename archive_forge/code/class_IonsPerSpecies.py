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
class IonsPerSpecies(SimpleVaspHeaderParser):
    """Example line:

    "   ions per type =              32  31   2"
    """
    LINE_DELIMITER = 'ions per type'

    def parse(self, cursor: _CURSOR, lines: _CHUNK) -> _RESULT:
        line = lines[cursor].strip()
        parts = line.split()
        ion_types = list(map(int, parts[4:]))
        return {'ion_types': ion_types}