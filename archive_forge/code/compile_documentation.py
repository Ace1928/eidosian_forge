from functools import partial
from typing import Sequence, Callable
from pennylane.queuing import QueuingManager
from pennylane.ops import __all__ as all_ops
from pennylane.tape import QuantumTape
from pennylane.transforms.core import transform, TransformDispatcher
from pennylane.transforms.optimization import (
A postprocesing function returned by a transform that only converts the batch of results
        into a result for a single ``QuantumTape``.
        