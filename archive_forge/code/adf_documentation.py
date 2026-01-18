from __future__ import annotations
import os
import re
from typing import TYPE_CHECKING
from monty.io import reverse_readline
from monty.itertools import chunks
from monty.json import MSONable
from monty.serialization import zopen
from pymatgen.core.structure import Molecule
A generator form of s.split('\n') for reducing memory overhead.

            Args:
                s (str): A multi-line string.

            Yields:
                str: line
            