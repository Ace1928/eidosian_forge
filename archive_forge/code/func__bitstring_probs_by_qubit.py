import sys
from collections import namedtuple
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple, TYPE_CHECKING, Union, cast
import numpy as np
from pyquil.external.rpcq import CompilerISA
from pyquil.gates import I, RX, MEASURE
from pyquil.noise_gates import _get_qvm_noise_supported_gates
from pyquil.quilatom import MemoryReference, format_parameter, ParameterDesignator
from pyquil.quilbase import Pragma, Gate, Declare
def _bitstring_probs_by_qubit(p: np.ndarray) -> np.ndarray:
    """
    Ensure that an array ``p`` with bitstring probabilities has a separate axis for each qubit such
    that ``p[i,j,...,k]`` gives the estimated probability of bitstring ``ij...k``.

    This should not allocate much memory if ``p`` is already in ``C``-contiguous order (row-major).

    :param p: An array that enumerates bitstring probabilities. When flattened out
        ``p = [p_00...0, p_00...1, ...,p_11...1]``. The total number of elements must therefore be a
        power of 2.
    :return: A reshaped view of ``p`` with a separate length-2 axis for each bit.
    """
    p = np.asarray(p, order='C')
    num_qubits = int(round(np.log2(p.size)))
    return p.reshape((2,) * num_qubits)