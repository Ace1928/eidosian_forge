import enum
from typing import Any, List, Optional, TYPE_CHECKING, Union
import pandas as pd
import sympy
from matplotlib import pyplot as plt
from cirq import circuits, ops, study, value
from cirq._compat import proper_repr
class ExperimentType(enum.Enum):
    RAMSEY = 1
    HAHN_ECHO = 2
    CPMG = 3