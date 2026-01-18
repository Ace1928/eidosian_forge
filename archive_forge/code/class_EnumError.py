from __future__ import annotations
import fractions
import itertools
import logging
import math
import re
import subprocess
from glob import glob
from shutil import which
from threading import Timer
import numpy as np
from monty.dev import requires
from monty.fractions import lcm
from monty.tempfile import ScratchDir
from pymatgen.core import DummySpecies, PeriodicSite, Structure
from pymatgen.io.vasp.inputs import Poscar
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
class EnumError(BaseException):
    """Error subclass for enumeration errors."""