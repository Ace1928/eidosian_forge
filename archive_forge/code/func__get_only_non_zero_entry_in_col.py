from ..pari import pari
import fractions
def _get_only_non_zero_entry_in_col(m, col):
    entry = None
    for row in m:
        if not row[col] == 0:
            assert entry is None, 'more than one non-zero entry in column %d' % col
            entry = row[col]
    if entry is not None:
        return entry
    return 0