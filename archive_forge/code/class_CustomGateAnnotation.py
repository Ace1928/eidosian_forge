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
class CustomGateAnnotation(cirq.Gate):

    def __init__(self, text: str):
        self.text = text

    def _num_qubits_(self):
        return 0

    def _circuit_diagram_info_(self, args) -> str:
        return self.text