import itertools
import pennylane as qml
from .matrices import core_matrix, mol_density_matrix, overlap_matrix, repulsion_tensor
def _scf(*args):
    """Perform the self-consistent-field iterations.

        Args:
            *args (array[array[float]]): initial values of the differentiable parameters

        Returns:
            tuple(array[float]): eigenvalues of the Fock matrix, molecular orbital coefficients,
            Fock matrix, core matrix
        """
    basis_functions = mol.basis_set
    charges = mol.nuclear_charges
    r = mol.coordinates
    n_electron = mol.n_electrons
    if r.requires_grad:
        args_r = [[args[0][i]] * mol.n_basis[i] for i in range(len(mol.n_basis))]
        args_ = [*args] + [qml.math.vstack(list(itertools.chain(*args_r)))]
        rep_tensor = repulsion_tensor(basis_functions)(*args_[1:])
        s = overlap_matrix(basis_functions)(*args_[1:])
        h_core = core_matrix(basis_functions, charges, r)(*args_)
    else:
        rep_tensor = repulsion_tensor(basis_functions)(*args)
        s = overlap_matrix(basis_functions)(*args)
        h_core = core_matrix(basis_functions, charges, r)(*args)
    rng = qml.math.random.default_rng(2030)
    s = s + qml.math.diag(rng.random(len(s)) * 1e-12)
    w, v = qml.math.linalg.eigh(s)
    x = v @ qml.math.diag(1.0 / qml.math.sqrt(w)) @ v.T
    eigvals, w_fock = qml.math.linalg.eigh(x.T @ h_core @ x)
    coeffs = x @ w_fock
    p = mol_density_matrix(n_electron, coeffs)
    for _ in range(n_steps):
        j = qml.math.einsum('pqrs,rs->pq', rep_tensor, p)
        k = qml.math.einsum('psqr,rs->pq', rep_tensor, p)
        fock_matrix = h_core + 2 * j - k
        eigvals, w_fock = qml.math.linalg.eigh(x.T @ fock_matrix @ x)
        coeffs = x @ w_fock
        p_update = mol_density_matrix(n_electron, coeffs)
        if qml.math.linalg.norm(p_update - p) <= tol:
            break
        p = p_update
    mol.mo_coefficients = coeffs
    return (eigvals, coeffs, fock_matrix, h_core, rep_tensor)