import difflib
import numpy as np
import os
import re
import glob
import shutil
import sys
import json
import time
import tempfile
import warnings
import subprocess
from copy import deepcopy
from collections import namedtuple
from itertools import product
from typing import List, Set
import ase
import ase.units as units
from ase.calculators.general import Calculator
from ase.calculators.calculator import compare_atoms
from ase.calculators.calculator import PropertyNotImplementedError
from ase.calculators.calculator import kpts2sizeandoffsets
from ase.dft.kpoints import BandPath
from ase.parallel import paropen
from ase.io.castep import read_param
from ase.io.castep import read_bands
from ase.constraints import FixCartesian
class CastepOptionDict:
    """A dictionary-like object to hold a set of options for .cell or .param
    files loaded from a dictionary, for the sake of validation.

    Replaces the old CastepCellDict and CastepParamDict that were defined in
    the castep_keywords.py file.
    """

    def __init__(self, options=None):
        object.__init__(self)
        self._options = {}
        for kw in options:
            opt = CastepOption(**options[kw])
            self._options[opt.keyword] = opt
            self.__dict__[opt.keyword] = opt