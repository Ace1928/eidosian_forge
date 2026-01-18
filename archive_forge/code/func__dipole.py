import pennylane as qml
from pennylane.fermi import FermiSentence, FermiWord
from .basis_data import atomic_numbers
from .hartree_fock import scf
from .matrices import moment_matrix
from .observable_hf import fermionic_observable, qubit_observable
def _dipole(*args):
    """Compute the qubit dipole moment observable.

        Args:
            *args (array[array[float]]): initial values of the differentiable parameters

        Returns:
            (list[Hamiltonian]): x, y and z components of the dipole moment observable
        """
    d = []
    d_ferm = fermionic_dipole(mol, cutoff, core, active)(*args)
    for i in d_ferm:
        d.append(qubit_observable(i, cutoff=cutoff))
    return d