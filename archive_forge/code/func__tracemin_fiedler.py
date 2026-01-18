from functools import partial
import networkx as nx
from networkx.utils import (
def _tracemin_fiedler(L, X, normalized, tol, method):
    """Compute the Fiedler vector of L using the TraceMIN-Fiedler algorithm.

    The Fiedler vector of a connected undirected graph is the eigenvector
    corresponding to the second smallest eigenvalue of the Laplacian matrix
    of the graph. This function starts with the Laplacian L, not the Graph.

    Parameters
    ----------
    L : Laplacian of a possibly weighted or normalized, but undirected graph

    X : Initial guess for a solution. Usually a matrix of random numbers.
        This function allows more than one column in X to identify more than
        one eigenvector if desired.

    normalized : bool
        Whether the normalized Laplacian matrix is used.

    tol : float
        Tolerance of relative residual in eigenvalue computation.
        Warning: There is no limit on number of iterations.

    method : string
        Should be 'tracemin_pcg' or 'tracemin_lu'.
        Otherwise exception is raised.

    Returns
    -------
    sigma, X : Two NumPy arrays of floats.
        The lowest eigenvalues and corresponding eigenvectors of L.
        The size of input X determines the size of these outputs.
        As this is for Fiedler vectors, the zero eigenvalue (and
        constant eigenvector) are avoided.
    """
    import numpy as np
    import scipy as sp
    n = X.shape[0]
    if normalized:
        e = np.sqrt(L.diagonal())
        D = sp.sparse.csr_array(sp.sparse.spdiags(1 / e, 0, n, n, format='csr'))
        L = D @ L @ D
        e *= 1.0 / np.linalg.norm(e, 2)
    if normalized:

        def project(X):
            """Make X orthogonal to the nullspace of L."""
            X = np.asarray(X)
            for j in range(X.shape[1]):
                X[:, j] -= X[:, j] @ e * e
    else:

        def project(X):
            """Make X orthogonal to the nullspace of L."""
            X = np.asarray(X)
            for j in range(X.shape[1]):
                X[:, j] -= X[:, j].sum() / n
    if method == 'tracemin_pcg':
        D = L.diagonal().astype(float)
        solver = _PCGSolver(lambda x: L @ x, lambda x: D * x)
    elif method == 'tracemin_lu':
        A = sp.sparse.csc_array(L, dtype=float, copy=True)
        i = (A.indptr[1:] - A.indptr[:-1]).argmax()
        A[i, i] = float('inf')
        solver = _LUSolver(A)
    else:
        raise nx.NetworkXError(f'Unknown linear system solver: {method}')
    Lnorm = abs(L).sum(axis=1).flatten().max()
    project(X)
    W = np.ndarray(X.shape, order='F')
    while True:
        X = np.linalg.qr(X)[0]
        W[:, :] = L @ X
        H = X.T @ W
        sigma, Y = sp.linalg.eigh(H, overwrite_a=True)
        X = X @ Y
        res = sp.linalg.blas.dasum(W @ Y[:, 0] - sigma[0] * X[:, 0]) / Lnorm
        if res < tol:
            break
        W[:, :] = solver.solve(X, tol)
        X = (sp.linalg.inv(W.T @ X) @ W.T).T
        project(X)
    return (sigma, np.asarray(X))