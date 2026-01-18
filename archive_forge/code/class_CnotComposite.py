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
class CnotComposite(cirq.testing.TwoQubitGate):

    def _decompose_(self, qubits):
        q0, q1 = qubits
        return (cirq.Y(q1) ** (-0.5), cirq.CZ(q0, q1), cirq.Y(q1) ** 0.5)