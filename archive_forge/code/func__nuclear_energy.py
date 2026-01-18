import itertools
import pennylane as qml
from .matrices import core_matrix, mol_density_matrix, overlap_matrix, repulsion_tensor
def _nuclear_energy(*args):
    """Compute the nuclear-repulsion energy.

        Args:
            *args (array[array[float]]): initial values of the differentiable parameters

        Returns:
            array[float]: nuclear-repulsion energy
        """
    if r.requires_grad:
        coor = args[0]
    else:
        coor = r
    e = qml.math.array([0.0])
    for i, r1 in enumerate(coor):
        for j, r2 in enumerate(coor[i + 1:]):
            e = e + charges[i] * charges[i + j + 1] / qml.math.linalg.norm(r1 - r2)
    return e