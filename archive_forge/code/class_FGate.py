import itertools
import os
import time
from collections import defaultdict
from random import randint, random, sample, randrange
from typing import Iterator, Optional, Tuple, TYPE_CHECKING
import numpy as np
import pytest
import sympy
import cirq
from cirq import circuits
from cirq import ops
from cirq.testing.devices import ValidatingTestDevice
class FGate(cirq.Gate):

    def __init__(self, num_qubits=1):
        self._num_qubits = num_qubits

    def num_qubits(self) -> int:
        return self._num_qubits

    def __repr__(self):
        return 'python-object-FGate:arbitrary-digits'