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
class SimpleProperty(VaspPropertyParser, ABC):
    LINE_DELIMITER = None

    def __init__(self):
        super().__init__()
        if self.LINE_DELIMITER is None:
            raise ValueError('Must specify a line delimiter.')

    def has_property(self, cursor, lines) -> bool:
        line = lines[cursor]
        return self.LINE_DELIMITER in line