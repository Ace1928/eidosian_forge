from __future__ import annotations
import codecs
import contextlib
import hashlib
import itertools
import json
import logging
import math
import os
import re
import subprocess
import warnings
from collections import namedtuple
from enum import Enum, unique
from glob import glob
from hashlib import sha256
from typing import TYPE_CHECKING, Any, Literal, cast
import numpy as np
import scipy.constants as const
from monty.io import zopen
from monty.json import MontyDecoder, MSONable
from monty.os import cd
from monty.os.path import zpath
from monty.serialization import dumpfn, loadfn
from tabulate import tabulate
from pymatgen.core import SETTINGS, Element, Lattice, Structure, get_el_sp
from pymatgen.electronic_structure.core import Magmom
from pymatgen.util.io_utils import clean_lines
from pymatgen.util.string import str_delimited
@unique
class KpointsSupportedModes(Enum):
    """Enum type of all supported modes for Kpoint generation."""
    Automatic = 0
    Gamma = 1
    Monkhorst = 2
    Line_mode = 3
    Cartesian = 4
    Reciprocal = 5

    def __str__(self):
        return str(self.name)

    @classmethod
    def from_str(cls, mode: str) -> Self:
        """
        Args:
            mode: String

        Returns:
            Kpoints_supported_modes
        """
        initial = mode.lower()[0]
        for key in cls:
            if key.name.lower()[0] == initial:
                return key
        raise ValueError(f'Invalid Kpoint mode={mode!r}')