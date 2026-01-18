import re
from Bio.SearchIO._index import SearchIndexer
from Bio.SearchIO._model import QueryResult, Hit, HSP, HSPFragment
def _get_frag_strand(self, frag, seq_type, parsedict):
    """Return fragment strand for given object (PRIVATE).

        Returns ``HSPFragment`` strand given the object, its sequence type,
        and its parsed dictionary values.
        """
    assert seq_type in ('query', 'hit')
    strand = getattr(frag, '%s_strand' % seq_type, None)
    if strand is not None:
        return strand
    else:
        start = parsedict.get('%s_start' % seq_type)
        end = parsedict.get('%s_end' % seq_type)
        if start is not None and end is not None:
            return 1 if start <= end else -1