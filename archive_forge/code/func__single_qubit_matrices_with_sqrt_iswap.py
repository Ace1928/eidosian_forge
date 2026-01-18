from typing import Optional, Sequence, Tuple, TYPE_CHECKING
import numpy as np
import sympy
from cirq import circuits, ops, linalg, protocols
from cirq.transformers.analytical_decompositions import single_qubit_decompositions
from cirq.transformers.merge_single_qubit_gates import merge_single_qubit_gates_to_phxz
def _single_qubit_matrices_with_sqrt_iswap(kak: 'cirq.KakDecomposition', required_sqrt_iswap_count: Optional[int]=None, atol: float=1e-08) -> Tuple[Sequence[Tuple[np.ndarray, np.ndarray]], complex]:
    """Computes the sequence of interleaved single-qubit unitary matrices in the
    sqrt-iSWAP decomposition."""
    decomposers = [(_in_0_region, _decomp_0_matrices), (_in_1sqrt_iswap_region, _decomp_1sqrt_iswap_matrices), (_in_2sqrt_iswap_region, _decomp_2sqrt_iswap_matrices), (_in_3sqrt_iswap_region, _decomp_3sqrt_iswap_matrices)]
    if required_sqrt_iswap_count is not None:
        if not 0 <= required_sqrt_iswap_count <= 3:
            raise ValueError('the argument `required_sqrt_iswap_count` must be 0, 1, 2, or 3.')
        can_decompose, decomposer = decomposers[required_sqrt_iswap_count]
        if not can_decompose(kak.interaction_coefficients, weyl_tol=atol / 10):
            raise ValueError(f'the given gate cannot be decomposed into exactly {required_sqrt_iswap_count} sqrt-iSWAP gates.')
        return decomposer(kak, atol=atol)
    for can_decompose, decomposer in decomposers:
        if can_decompose(kak.interaction_coefficients, weyl_tol=atol / 10):
            return decomposer(kak, atol)
    assert False, 'The final can_decompose should always returns True'