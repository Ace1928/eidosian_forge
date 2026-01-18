import networkx as nx
import numpy as np
from scipy.sparse import linalg
from . import _ncut, _ncut_cy
def _ncut_relabel(rag, thresh, num_cuts, random_generator):
    """Perform Normalized Graph cut on the Region Adjacency Graph.

    Recursively partition the graph into 2, until further subdivision
    yields a cut greater than `thresh` or such a cut cannot be computed.
    For such a subgraph, indices to labels of all its nodes map to a single
    unique value.

    Parameters
    ----------
    rag : RAG
        The region adjacency graph.
    thresh : float
        The threshold. A subgraph won't be further subdivided if the
        value of the N-cut exceeds `thresh`.
    num_cuts : int
        The number or N-cuts to perform before determining the optimal one.
    random_generator : `numpy.random.Generator`
        Provides initial values for eigenvalue solver.
    """
    d, w = _ncut.DW_matrices(rag)
    m = w.shape[0]
    if m > 2:
        d2 = d.copy()
        d2.data = np.reciprocal(np.sqrt(d2.data, out=d2.data), out=d2.data)
        A = d2 * (d - w) * d2
        v0 = random_generator.random(A.shape[0])
        vals, vectors = linalg.eigsh(A, which='SM', v0=v0, k=min(100, m - 2))
        vals, vectors = (np.real(vals), np.real(vectors))
        index2 = _ncut_cy.argmin2(vals)
        ev = vectors[:, index2]
        cut_mask, mcut = get_min_ncut(ev, d, w, num_cuts)
        if mcut < thresh:
            sub1, sub2 = partition_by_cut(cut_mask, rag)
            _ncut_relabel(sub1, thresh, num_cuts, random_generator)
            _ncut_relabel(sub2, thresh, num_cuts, random_generator)
            return
    _label_all(rag, 'ncut label')