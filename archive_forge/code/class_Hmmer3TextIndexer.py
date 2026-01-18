import re
from Bio.SearchIO._utils import read_forward, removesuffix
from Bio.SearchIO._model import QueryResult, Hit, HSP, HSPFragment
from ._base import _BaseHmmerTextIndexer
class Hmmer3TextIndexer(_BaseHmmerTextIndexer):
    """Indexer class for HMMER plain text output."""
    _parser = Hmmer3TextParser
    qresult_start = b'Query: '
    qresult_end = b'//'

    def __iter__(self):
        """Iterate over Hmmer3TextIndexer; yields query results' key, offsets, 0."""
        handle = self._handle
        handle.seek(0)
        start_offset = handle.tell()
        regex_id = re.compile(_QRE_ID_LEN_PTN.encode())
        while True:
            line = read_forward(handle)
            end_offset = handle.tell()
            if line.startswith(self.qresult_start):
                regx = re.search(regex_id, line)
                qresult_key = regx.group(1).strip()
                start_offset = end_offset - len(line)
            elif line.startswith(self.qresult_end):
                yield (qresult_key.decode(), start_offset, 0)
                start_offset = end_offset
            elif not line:
                break