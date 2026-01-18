from typing import Sequence, Callable
import pennylane as qml
from .core import transform
A postprocesing function returned by a transform that only converts the batch of results
            into a result for a single ``QuantumTape``.
            