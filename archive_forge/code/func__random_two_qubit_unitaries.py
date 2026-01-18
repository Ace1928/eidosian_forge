import random
import numpy as np
import pytest
import cirq
from cirq import value
from cirq import unitary_eig
def _random_two_qubit_unitaries(num_samples: int, random_state: 'cirq.RANDOM_STATE_OR_SEED_LIKE'):
    kl = _local_two_qubit_unitaries(num_samples, random_state)
    kr = _local_two_qubit_unitaries(num_samples, random_state)
    prng = value.parse_random_state(random_state)
    kak_vecs = prng.rand(num_samples, 3) * np.pi
    gens = np.einsum('...a,abc->...bc', kak_vecs, _kak_gens)
    evals, evecs = np.linalg.eigh(gens)
    A = np.einsum('...ab,...b,...cb', evecs, np.exp(1j * evals), evecs.conj())
    return (np.einsum('...ab,...bc,...cd', kl, A, kr), kak_vecs)