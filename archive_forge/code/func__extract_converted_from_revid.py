from collections import defaultdict
from breezy import errors, foreign, ui, urlutils
def _extract_converted_from_revid(rev):
    if 'converted-from' not in rev.properties:
        return
    for line in rev.properties.get('converted-from', '').splitlines():
        kind, serialized_foreign_revid = line.split(' ', 1)
        yield (kind, serialized_foreign_revid)