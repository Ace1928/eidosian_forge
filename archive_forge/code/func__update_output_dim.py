import contextlib
import copy
from collections import Counter
from typing import List, Union, Optional, Sequence
import pennylane as qml
from pennylane.measurements import (
from pennylane.typing import TensorLike
from pennylane.operation import Observable, Operator, Operation, _UNSET_BATCH_SIZE
from pennylane.pytrees import register_pytree
from pennylane.queuing import AnnotatedQueue, process_queue
from pennylane.wires import Wires
def _update_output_dim(self):
    """Update the dimension of the output of the quantum script.

        Sets:
            self._output_dim (int): Size of the quantum script output (when flattened)

        This method makes use of `self.batch_size`, so that `self._batch_size`
        needs to be up to date when calling it.
        Call `_update_batch_size` before `_update_output_dim`
        """
    self._output_dim = 0
    for m in self.measurements:
        if isinstance(m, ProbabilityMP):
            self._output_dim += 2 ** len(m.wires)
        elif not isinstance(m, StateMP):
            self._output_dim += 1
    if self.batch_size:
        self._output_dim *= self.batch_size