from typing import AbstractSet, Iterator, Any
import pytest
import numpy as np
import sympy
import cirq
class NotImplementedGate3(cirq.Gate):

    def _qid_shape_(self):
        return NotImplemented