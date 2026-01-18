import re
from Bio.SearchIO._index import SearchIndexer
from Bio.SearchIO._model import QueryResult, Hit, HSP, HSPFragment
def _adjust_coords(self, field, value, hsp):
    """Adjust start and end coordinates according to strand (PRIVATE)."""
    assert field in ('qstart', 'qend', 'sstart', 'send')
    seq_type = 'query' if field.startswith('q') else 'hit'
    strand = getattr(hsp, '%s_strand' % seq_type, None)
    if strand is None:
        raise ValueError('Required attribute %r not found.' % ('%s_strand' % seq_type))
    if strand < 0:
        if field.endswith('start'):
            value = getattr(hsp, '%s_end' % seq_type)
        elif field.endswith('end'):
            value = getattr(hsp, '%s_start' % seq_type) + 1
    elif field.endswith('start'):
        value += 1
    return value