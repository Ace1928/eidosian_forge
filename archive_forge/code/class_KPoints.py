import os
import copy
import subprocess
from math import pi, sqrt
import pathlib
from typing import Union, Optional, List, Set, Dict, Any
import warnings
import numpy as np
from ase.cell import Cell
from ase.outputs import Properties, all_outputs
from ase.utils import jsonable
from ase.calculators.abc import GetPropertiesMixin
@jsonable('kpoints')
class KPoints:

    def __init__(self, kpts=None):
        if kpts is None:
            kpts = np.zeros((1, 3))
        self.kpts = kpts

    def todict(self):
        return vars(self)