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
class PropertyNotPresent(CalculatorError):
    """Requested property is missing.

    Maybe it was never calculated, or for some reason was not extracted
    with the rest of the results, without being a fatal ReadError."""