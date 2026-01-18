import sys
from collections import namedtuple
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple, TYPE_CHECKING, Union, cast
import numpy as np
from pyquil.external.rpcq import CompilerISA
from pyquil.gates import I, RX, MEASURE
from pyquil.noise_gates import _get_qvm_noise_supported_gates
from pyquil.quilatom import MemoryReference, format_parameter, ParameterDesignator
from pyquil.quilbase import Pragma, Gate, Declare
def _check_kraus_ops(n: int, kraus_ops: Sequence[np.ndarray]) -> None:
    """
    Verify that the Kraus operators are of the correct shape and satisfy the correct normalization.

    :param n: Number of qubits
    :param kraus_ops: The Kraus operators as numpy.ndarrays.
    """
    for k in kraus_ops:
        if not np.shape(k) == (2 ** n, 2 ** n):
            raise ValueError('Kraus operators for {0} qubits must have shape {1}x{1}: {2}'.format(n, 2 ** n, k))
    kdk_sum = sum((np.transpose(k).conjugate().dot(k) for k in kraus_ops))
    if not np.allclose(kdk_sum, np.eye(2 ** n), atol=0.001):
        raise ValueError('Kraus operator not correctly normalized: sum_j K_j^*K_j == {}'.format(kdk_sum))