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
class PositionsAndForces(SimpleVaspChunkParser):
    """Positions and forces are written in the same block.
    We parse both simultaneously"""
    LINE_DELIMITER = 'POSITION          '

    def parse(self, cursor: _CURSOR, lines: _CHUNK) -> _RESULT:
        nskip = 2
        natoms = self.get_from_header('natoms')
        positions = np.zeros((natoms, 3))
        forces = np.zeros((natoms, 3))
        for i in range(natoms):
            line = self.get_line(cursor + i + nskip, lines)
            parts = list(map(float, line.split()))
            positions[i] = parts[0:3]
            forces[i] = parts[3:6]
        return {'positions': positions, 'forces': forces}