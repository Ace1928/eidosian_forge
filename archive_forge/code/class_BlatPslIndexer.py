import re
from math import log
from Bio.SearchIO._index import SearchIndexer
from Bio.SearchIO._model import QueryResult, Hit, HSP, HSPFragment
class BlatPslIndexer(SearchIndexer):
    """Indexer class for BLAT PSL output."""
    _parser = BlatPslParser

    def __init__(self, filename, pslx=False):
        """Initialize the class."""
        SearchIndexer.__init__(self, filename, pslx=pslx)

    def __iter__(self):
        """Iterate over the file handle; yields key, start offset, and length."""
        handle = self._handle
        handle.seek(0)
        query_id_idx = 9
        qresult_key = None
        tab_char = b'\t'
        start_offset = handle.tell()
        line = handle.readline()
        while not re.search(_RE_ROW_CHECK_IDX, line.strip()):
            start_offset = handle.tell()
            line = handle.readline()
            if not line:
                return
        while True:
            end_offset = handle.tell()
            cols = [x for x in line.strip().split(tab_char) if x]
            if qresult_key is None:
                qresult_key = cols[query_id_idx]
            else:
                curr_key = cols[query_id_idx]
                if curr_key != qresult_key:
                    yield (qresult_key.decode(), start_offset, end_offset - start_offset)
                    qresult_key = curr_key
                    start_offset = end_offset - len(line)
            line = handle.readline()
            if not line:
                yield (qresult_key.decode(), start_offset, end_offset - start_offset)
                break

    def get_raw(self, offset):
        """Return raw bytes string of a QueryResult object from the given offset."""
        handle = self._handle
        handle.seek(offset)
        query_id_idx = 9
        qresult_key = None
        qresult_raw = b''
        tab_char = b'\t'
        while True:
            line = handle.readline()
            if not line:
                break
            cols = [x for x in line.strip().split(tab_char) if x]
            if qresult_key is None:
                qresult_key = cols[query_id_idx]
            else:
                curr_key = cols[query_id_idx]
                if curr_key != qresult_key:
                    break
            qresult_raw += line
        return qresult_raw