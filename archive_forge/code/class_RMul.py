import itertools
import math
from typing import List
import numpy as np
import pytest
import sympy
import cirq
import cirq.testing
class RMul:

    def __rmul__(self, other):
        return 'Yay!'