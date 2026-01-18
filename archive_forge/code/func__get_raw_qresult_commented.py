import re
from Bio.SearchIO._index import SearchIndexer
from Bio.SearchIO._model import QueryResult, Hit, HSP, HSPFragment
def _get_raw_qresult_commented(self, offset):
    """Return the bytes raw string of a single QueryResult from a commented file (PRIVATE)."""
    handle = self._handle
    handle.seek(offset)
    qresult_raw = b''
    end_mark = b'# BLAST processed'
    query_mark = None
    line = handle.readline()
    while line:
        if query_mark is None:
            query_mark = line
        elif line == query_mark or line.startswith(end_mark):
            break
        qresult_raw += line
        line = handle.readline()
    return qresult_raw