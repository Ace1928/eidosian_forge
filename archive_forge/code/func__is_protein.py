import re
from math import log
from Bio.SearchIO._index import SearchIndexer
from Bio.SearchIO._model import QueryResult, Hit, HSP, HSPFragment
def _is_protein(psl):
    """Validate if psl is protein (PRIVATE)."""
    if len(psl['strand']) == 2:
        if psl['strand'][1] == '+':
            return psl['tend'] == psl['tstarts'][-1] + 3 * psl['blocksizes'][-1]
        elif psl['strand'][1] == '-':
            return psl['tstart'] == psl['tsize'] - (psl['tstarts'][-1] + 3 * psl['blocksizes'][-1])
    return False