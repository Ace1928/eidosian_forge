import math
from typing import Any, Dict, List, Sequence, Tuple
import numpy as np
import pytest
import sympy
import cirq
class CountingTrialResult(cirq.SimulationTrialResultBase[CountingSimulationState]):
    pass