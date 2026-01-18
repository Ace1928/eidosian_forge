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
class MultiTargetCZ(cirq.Gate):

    def __init__(self, num_qubits):
        self._num_qubits = num_qubits

    def num_qubits(self) -> int:
        return self._num_qubits

    def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs) -> Tuple[str, ...]:
        assert args.known_qubit_count is not None
        return ('@',) + ('Z',) * (args.known_qubit_count - 1)