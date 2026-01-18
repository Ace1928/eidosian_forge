from typing import List, Tuple
import re
import numpy as np
import pytest
import sympy
import cirq
from cirq import value
from cirq.testing import assert_has_consistent_trace_distance_bound
Two-qubit gate for the following matrix:
    [1  0  0  0]
    [0  1  0  0]
    [0  0  i  0]
    [0  0  0 -i]
    