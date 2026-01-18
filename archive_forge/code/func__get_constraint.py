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
def _get_constraint(self):
    """Try and get the constraints from the POSCAR of CONTCAR
        since they aren't located in the OUTCAR, and thus we cannot construct an
        OUTCAR parser which does this.
        """
    constraint = None
    if self.workdir is not None:
        constraint = read_constraints_from_file(self.workdir)
    return constraint