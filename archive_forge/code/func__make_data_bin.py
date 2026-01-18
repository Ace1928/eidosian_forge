from __future__ import annotations
from abc import abstractmethod, ABC
from collections.abc import Iterable, Sequence
from copy import copy
from typing import Generic, TypeVar
import numpy as np
from numpy.typing import NDArray
from qiskit.circuit import QuantumCircuit
from qiskit.providers import JobV1 as Job
from qiskit.quantum_info.operators import SparsePauliOp
from qiskit.quantum_info.operators.base_operator import BaseOperator
from ..containers import (
from ..containers.estimator_pub import EstimatorPub
from . import validation
from .base_primitive import BasePrimitive
from .base_primitive_job import BasePrimitiveJob
@staticmethod
def _make_data_bin(pub: EstimatorPub) -> DataBin:
    return make_data_bin((('evs', NDArray[np.float64]), ('stds', NDArray[np.float64])), pub.shape)