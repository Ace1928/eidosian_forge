from . import matrix
def _enumerate_from_basis(basis, l, N):
    """
    Given a list of pairs `(v_i, o_i)` where each `v_i` is a vector of length l
    and an integer N > 2, iterate through all linear combinations
    `k_0 v_0 + k_1 v_1 + ... + k_m v_m` (modulo N) where `k_i = 0, ..., o_i`.

    If basis is empty, just return zero vector of length l.
    """
    if len(basis) == 0:
        yield (l * [0])
    else:
        basis_vector, order = basis[0]
        for i in range(order):
            for vector in _enumerate_from_basis(basis[1:], l, N):
                yield [(i * b + v) % N for b, v in zip(basis_vector, vector)]