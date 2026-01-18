import math
from typing import Any, Dict, List, Sequence, Tuple
import numpy as np
import pytest
import sympy
import cirq
class SplittableCountingSimulationState(CountingSimulationState):

    @property
    def allows_factoring(self):
        return True