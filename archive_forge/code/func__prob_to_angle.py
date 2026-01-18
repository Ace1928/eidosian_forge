import math
from typing import Any, cast, Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING, Union
from sympy.combinatorics import GrayCode
from cirq import value
from cirq.ops import common_gates, pauli_gates, raw_types
def _prob_to_angle(prob):
    return 2.0 * math.asin(math.sqrt(prob))