from typing import Union, Tuple, cast
import numpy as np
import pytest
import sympy
import cirq
from cirq.type_workarounds import NotImplementedType
class UnphaseableGate(cirq.Gate):
    pass