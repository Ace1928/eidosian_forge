import itertools
from collections import defaultdict
import networkx as nx
from networkx.utils import not_implemented_for
def _csr_gen_triples(A):
    """Converts a SciPy sparse array in **Compressed Sparse Row** format to
    an iterable of weighted edge triples.

    """
    nrows = A.shape[0]
    data, indices, indptr = (A.data, A.indices, A.indptr)
    for i in range(nrows):
        for j in range(indptr[i], indptr[i + 1]):
            yield (i, int(indices[j]), data[j])