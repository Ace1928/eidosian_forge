import re
from Bio.SearchIO._index import SearchIndexer
from Bio.SearchIO._model import QueryResult, Hit, HSP, HSPFragment
def _set_qresult_hits(qresult, hit_rows=()):
    """Append Hits without alignments into QueryResults (PRIVATE)."""
    for hit_row in hit_rows:
        hit_id, remainder = hit_row.split(' ', 1)
        if hit_id not in qresult:
            frag = HSPFragment(hit_id, qresult.id)
            hsp = HSP([frag])
            hit = Hit([hsp])
            qresult.append(hit)
    return qresult