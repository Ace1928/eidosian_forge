import re
from math import log
from Bio.SearchIO._index import SearchIndexer
from Bio.SearchIO._model import QueryResult, Hit, HSP, HSPFragment
def _reorient_starts(starts, blksizes, seqlen, strand):
    """Reorients block starts into the opposite strand's coordinates (PRIVATE).

    :param starts: start coordinates
    :type starts: list [int]
    :param blksizes: block sizes
    :type blksizes: list [int]
    :param seqlen: sequence length
    :type seqlen: int
    :param strand: sequence strand
    :type strand: int, choice of -1, 0, or 1

    """
    if len(starts) != len(blksizes):
        raise RuntimeError('Unequal start coordinates and block sizes list (%r vs %r)' % (len(starts), len(blksizes)))
    if strand >= 0:
        return starts
    else:
        return [seqlen - start - blksize for start, blksize in zip(starts, blksizes)]