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
class CustomOperationAnnotation(cirq.Operation):

    def __init__(self, text: str):
        self.text = text

    def with_qubits(self, *new_qubits):
        raise NotImplementedError()

    @property
    def qubits(self):
        return ()

    def _circuit_diagram_info_(self, args) -> str:
        return self.text