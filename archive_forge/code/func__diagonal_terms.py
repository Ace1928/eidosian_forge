import pennylane as qml
from pennylane.operation import Tensor
def _diagonal_terms(hamiltonian):
    """Checks if all terms in a Hamiltonian are products of diagonal Pauli gates
    (:class:`~.PauliZ` and :class:`~.Identity`).

    Args:
        hamiltonian (.Hamiltonian): The Hamiltonian being checked

    Returns:
        bool: ``True`` if all terms are products of diagonal Pauli gates, ``False`` otherwise
    """
    val = True
    for i in hamiltonian.ops:
        obs = i.obs if isinstance(i, Tensor) else [i]
        for j in obs:
            if j.name not in ('PauliZ', 'Identity'):
                val = False
                break
    return val