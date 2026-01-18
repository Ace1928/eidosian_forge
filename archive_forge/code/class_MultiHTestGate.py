import itertools
import random
from typing import Type
from unittest import mock
import numpy as np
import pytest
import sympy
import cirq
class MultiHTestGate(cirq.testing.TwoQubitGate):

    def _decompose_(self, qubits):
        return cirq.H.on_each(*qubits)