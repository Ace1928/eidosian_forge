import re
from Bio.SearchIO._index import SearchIndexer
from Bio.SearchIO._model import QueryResult, Hit, HSP, HSPFragment
def _compute_gapopen_num(hsp):
    """Return the number of gap openings in the given HSP (PRIVATE)."""
    gapopen = 0
    for seq_type in ('query', 'hit'):
        seq = str(getattr(hsp, seq_type).seq)
        gapopen += len(re.findall(_RE_GAPOPEN, seq))
    return gapopen