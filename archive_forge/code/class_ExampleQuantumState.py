from typing import Any, Sequence, Tuple
from typing_extensions import Self
import numpy as np
import pytest
import cirq
class ExampleQuantumState(cirq.QuantumStateRepresentation):

    def copy(self, deep_copy_buffers=True):
        pass

    def measure(self, axes, seed=None):
        pass