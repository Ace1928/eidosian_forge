import abc
import copy
import types
import warnings
from collections import OrderedDict
from collections.abc import Iterable, Sequence
from functools import lru_cache
import numpy as np
import pennylane as qml
from pennylane.measurements import (
from pennylane.operation import Observable, Operation, Tensor, Operator, StatePrepBase
from pennylane.ops import Hamiltonian, Sum
from pennylane.tape import QuantumScript, QuantumTape, expand_tape_state_prep
from pennylane.wires import WireError, Wires
from pennylane.queuing import QueuingManager
def _has_partitioned_shots(self):
    """Checks if the device was instructed to perform executions with partitioned shots.

        Returns:
            bool: whether or not shots are partitioned
        """
    return self._shot_vector is not None and (len(self._shot_vector) > 1 or self._shot_vector[0].copies > 1)