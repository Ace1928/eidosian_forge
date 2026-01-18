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
class OUTCARChunk(ImageChunk):
    """Container class for a chunk of the OUTCAR which consists of a
    self-contained SCF step, i.e. and image. Also contains the header_data
    """

    def __init__(self, lines: _CHUNK, header: _HEADER, parser: ChunkParser=None):
        super().__init__()
        self.lines = lines
        self.header = header
        self.parser = parser or OutcarChunkParser()

    def build(self):
        self.parser.header = self.header
        return self.parser.build(self.lines)