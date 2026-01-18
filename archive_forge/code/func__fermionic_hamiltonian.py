import pennylane as qml
from .hartree_fock import nuclear_energy, scf
from .observable_hf import fermionic_observable, qubit_observable
def _fermionic_hamiltonian(*args):
    """Compute the fermionic hamiltonian.

        Args:
            *args (array[array[float]]): initial values of the differentiable parameters

        Returns:
            FermiSentence: fermionic Hamiltonian
        """
    core_constant, one, two = electron_integrals(mol, core, active)(*args)
    return fermionic_observable(core_constant, one, two, cutoff)