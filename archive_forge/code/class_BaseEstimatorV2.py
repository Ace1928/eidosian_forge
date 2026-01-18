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
class BaseEstimatorV2(ABC):
    """Estimator V2 base class.

    An estimator estimates expectation values for provided quantum circuit and
    observable combinations.

    An Estimator implementation must treat the :meth:`.run` method ``precision=None``
    kwarg as using a default ``precision`` value.  The default value and methods to
    set it can be determined by the Estimator implementor.
    """

    @staticmethod
    def _make_data_bin(pub: EstimatorPub) -> DataBin:
        return make_data_bin((('evs', NDArray[np.float64]), ('stds', NDArray[np.float64])), pub.shape)

    @abstractmethod
    def run(self, pubs: Iterable[EstimatorPubLike], *, precision: float | None=None) -> BasePrimitiveJob[PrimitiveResult[PubResult]]:
        """Estimate expectation values for each provided pub (Primitive Unified Bloc).

        Args:
            pubs: An iterable of pub-like objects, such as tuples ``(circuit, observables)``
                  or ``(circuit, observables, parameter_values)``.
            precision: The target precision for expectation value estimates of each
                       run Estimator Pub that does not specify its own precision. If None
                       the estimator's default precision value will be used.

        Returns:
            A job object that contains results.
        """