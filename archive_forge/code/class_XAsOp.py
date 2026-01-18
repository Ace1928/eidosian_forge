import itertools
import random
from typing import Type
from unittest import mock
import numpy as np
import pytest
import sympy
import cirq
import cirq.testing
class XAsOp(cirq.Operation):

    def __init__(self, q):
        self.q = q

    @property
    def qubits(self):
        return (self.q,)

    def with_qubits(self, *new_qubits):
        raise NotImplementedError()

    def _kraus_(self):
        return cirq.kraus(cirq.X)