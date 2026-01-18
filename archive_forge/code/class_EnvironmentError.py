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
class EnvironmentError(CalculatorSetupError):
    """Raised if calculator is not properly set up with ASE.

    May be missing an executable or environment variables."""