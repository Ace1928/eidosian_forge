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
def _update_par_info(self):
    """Update the parameter information list. Each entry in the list with an operation and an index
        into that operation's data.

        Sets:
            _par_info (list): Parameter information
        """
    self._par_info = []
    for idx, op in enumerate(self.operations):
        self._par_info.extend(({'op': op, 'op_idx': idx, 'p_idx': i} for i, d in enumerate(op.data)))
    n_ops = len(self.operations)
    for idx, m in enumerate(self.measurements):
        if m.obs is not None:
            self._par_info.extend(({'op': m.obs, 'op_idx': idx + n_ops, 'p_idx': i} for i, d in enumerate(m.obs.data)))