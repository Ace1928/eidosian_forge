import re
from ._base import _BaseExonerateParser, _BaseExonerateIndexer, _STRAND_MAP
from typing import Type
class ExonerateVulgarIndexer(_BaseExonerateIndexer):
    """Indexer class for exonerate vulgar lines."""
    _parser: Type[_BaseExonerateParser] = ExonerateVulgarParser
    _query_mark = b'vulgar'

    def get_qresult_id(self, pos):
        """Return the query ID of the nearest vulgar line."""
        handle = self._handle
        handle.seek(pos)
        line = handle.readline()
        assert line.startswith(self._query_mark), line
        id = re.search(_RE_VULGAR, line.decode())
        return id.group(1)

    def get_raw(self, offset):
        """Return the raw bytes string of a QueryResult object from the given offset."""
        handle = self._handle
        handle.seek(offset)
        qresult_key = None
        qresult_raw = b''
        while True:
            line = handle.readline()
            if not line:
                break
            elif line.startswith(self._query_mark):
                cur_pos = handle.tell() - len(line)
                if qresult_key is None:
                    qresult_key = self.get_qresult_id(cur_pos)
                else:
                    curr_key = self.get_qresult_id(cur_pos)
                    if curr_key != qresult_key:
                        break
            qresult_raw += line
        return qresult_raw