import re
from functools import reduce
from abc import ABC, abstractmethod
from typing import Optional, Type
from Bio.SearchIO._index import SearchIndexer
from Bio.SearchIO._model import QueryResult, Hit, HSP, HSPFragment
from Bio.SeqUtils import seq1
class _BaseExonerateIndexer(SearchIndexer):
    """Indexer class for Exonerate plain text."""
    _parser: Optional[Type[_BaseExonerateParser]] = None
    _query_mark: Optional[bytes] = None

    def get_qresult_id(self, pos):
        raise NotImplementedError('Should be defined by subclass')

    def __iter__(self):
        """Iterate over the file handle; yields key, start offset, and length."""
        handle = self._handle
        handle.seek(0)
        qresult_key = None
        while True:
            start_offset = handle.tell()
            line = handle.readline()
            if line.startswith(self._query_mark):
                if qresult_key is None:
                    qresult_key = self.get_qresult_id(start_offset)
                    qresult_offset = start_offset
                else:
                    curr_key = self.get_qresult_id(start_offset)
                    if curr_key != qresult_key:
                        yield (qresult_key, qresult_offset, start_offset - qresult_offset)
                        qresult_key = curr_key
                        qresult_offset = start_offset
                        handle.seek(qresult_offset)
            elif not line:
                yield (qresult_key, qresult_offset, start_offset - qresult_offset)
                break