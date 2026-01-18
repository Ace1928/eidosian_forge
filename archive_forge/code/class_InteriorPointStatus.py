from pyomo.contrib.pynumero.interfaces.utils import (
import numpy as np
import logging
import time
from pyomo.contrib.pynumero.linalg.base import LinearSolverStatus
from pyomo.common.timing import HierarchicalTimer
import enum
class InteriorPointStatus(enum.Enum):
    optimal = 0
    error = 1