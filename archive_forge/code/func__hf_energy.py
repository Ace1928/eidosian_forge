import itertools
import pennylane as qml
from .matrices import core_matrix, mol_density_matrix, overlap_matrix, repulsion_tensor
def _hf_energy(*args):
    """Compute the Hartree-Fock energy.

        Args:
            *args (array[array[float]]): initial values of the differentiable parameters

        Returns:
            float: the Hartree-Fock energy
        """
    _, coeffs, fock_matrix, h_core, _ = scf(mol)(*args)
    e_rep = nuclear_energy(mol.nuclear_charges, mol.coordinates)(*args)
    e_elec = qml.math.einsum('pq,qp', fock_matrix + h_core, mol_density_matrix(mol.n_electrons, coeffs))
    energy = e_elec + e_rep
    return energy.reshape(())