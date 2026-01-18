import math
from typing import Any, Dict, List, Sequence, Tuple
import numpy as np
import pytest
import sympy
import cirq
class DensityCountingSimulator(CountingSimulator):

    def _can_be_in_run_prefix(self, val):
        return not cirq.is_measurement(val)