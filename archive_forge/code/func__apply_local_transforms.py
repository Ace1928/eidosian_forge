import sys
from collections import namedtuple
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple, TYPE_CHECKING, Union, cast
import numpy as np
from pyquil.external.rpcq import CompilerISA
from pyquil.gates import I, RX, MEASURE
from pyquil.noise_gates import _get_qvm_noise_supported_gates
from pyquil.quilatom import MemoryReference, format_parameter, ParameterDesignator
from pyquil.quilbase import Pragma, Gate, Declare
def _apply_local_transforms(p: np.ndarray, ts: Iterable[np.ndarray]) -> np.ndarray:
    """
    Given a 2d array of single shot results (outer axis iterates over shots, inner axis over bits)
    and a list of assignment probability matrices (one for each bit in the readout, ordered like
    the inner axis of results) apply local 2x2 matrices to each bit index.

    :param p: An array that enumerates a function indexed by
        bitstrings::

            f(ijk...) = p[i,j,k,...]

    :param ts: A sequence of 2x2 transform-matrices, one for each bit.
    :return: ``p_transformed`` an array with as many dimensions as there are bits with the result of
        contracting p along each axis by the corresponding bit transformation::

            p_transformed[ijk...] = f'(ijk...) = sum_lmn... ts[0][il] ts[1][jm] ts[2][kn] f(lmn...)
    """
    p_corrected = _bitstring_probs_by_qubit(p)
    nq = p_corrected.ndim
    for idx, trafo_idx in enumerate(ts):
        einsum_pat = 'ij,' + _CHARS[:idx] + 'j' + _CHARS[idx:nq - 1] + '->' + _CHARS[:idx] + 'i' + _CHARS[idx:nq - 1]
        p_corrected = np.einsum(einsum_pat, trafo_idx, p_corrected)
    return p_corrected