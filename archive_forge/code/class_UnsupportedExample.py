from typing import Optional, Sequence, Type
import pytest
import cirq
import sympy
import numpy as np
class UnsupportedExample(cirq.testing.TwoQubitGate):
    pass