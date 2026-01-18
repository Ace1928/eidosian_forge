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
class CalculationFailed(CalculatorError):
    """Calculation failed unexpectedly.

    Reasons to raise this error are:
      * Calculation did not converge
      * Calculation ran out of memory
      * Segmentation fault or other abnormal termination
      * Arithmetic trouble (singular matrices, NaN, ...)

    Typically raised during calculation."""