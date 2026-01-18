import re
from Bio.SearchIO._index import SearchIndexer
from Bio.SearchIO._model import QueryResult, Hit, HSP, HSPFragment
class BlastTabIndexer(SearchIndexer):
    """Indexer class for BLAST+ tab output."""
    _parser = BlastTabParser

    def __init__(self, filename, comments=False, fields=_DEFAULT_FIELDS):
        """Initialize the class."""
        SearchIndexer.__init__(self, filename, comments=comments, fields=fields)
        if not self._kwargs['comments']:
            if 'qseqid' in fields:
                self._key_idx = fields.index('qseqid')
            elif 'qacc' in fields:
                self._key_idx = fields.index('qacc')
            elif 'qaccver' in fields:
                self._key_idx = fields.index('qaccver')
            else:
                raise ValueError("Custom fields is missing an ID column. One of these must be present: 'qseqid', 'qacc', or 'qaccver'.")

    def __iter__(self):
        """Iterate over the file handle; yields key, start offset, and length."""
        handle = self._handle
        handle.seek(0)
        if not self._kwargs['comments']:
            iterfunc = self._qresult_index
        else:
            iterfunc = self._qresult_index_commented
        for key, offset, length in iterfunc():
            yield (key.decode(), offset, length)

    def _qresult_index_commented(self):
        """Indexer for commented BLAST tabular files (PRIVATE)."""
        handle = self._handle
        handle.seek(0)
        start_offset = 0
        query_mark = None
        qid_mark = b'# Query: '
        end_mark = b'# BLAST processed'
        while True:
            end_offset = handle.tell()
            line = handle.readline()
            if query_mark is None:
                query_mark = line
                start_offset = end_offset
            elif line.startswith(qid_mark):
                qresult_key = line[len(qid_mark):].split()[0]
            elif line == query_mark or line.startswith(end_mark):
                yield (qresult_key, start_offset, end_offset - start_offset)
                start_offset = end_offset
            elif not line:
                break

    def _qresult_index(self):
        """Indexer for noncommented BLAST tabular files (PRIVATE)."""
        handle = self._handle
        handle.seek(0)
        start_offset = 0
        qresult_key = None
        key_idx = self._key_idx
        while True:
            end_offset = handle.tell()
            line = handle.readline()
            if qresult_key is None:
                qresult_key = line.split(b'\t')[key_idx]
            else:
                try:
                    curr_key = line.split(b'\t')[key_idx]
                except IndexError:
                    curr_key = b''
                if curr_key != qresult_key:
                    yield (qresult_key, start_offset, end_offset - start_offset)
                    qresult_key = curr_key
                    start_offset = end_offset
            if not line:
                break

    def get_raw(self, offset):
        """Return the raw bytes string of a QueryResult object from the given offset."""
        if self._kwargs['comments']:
            getfunc = self._get_raw_qresult_commented
        else:
            getfunc = self._get_raw_qresult
        return getfunc(offset)

    def _get_raw_qresult(self, offset):
        """Return the raw bytes string of a single QueryResult from a noncommented file (PRIVATE)."""
        handle = self._handle
        handle.seek(offset)
        qresult_raw = b''
        key_idx = self._key_idx
        qresult_key = None
        while True:
            line = handle.readline()
            if qresult_key is None:
                qresult_key = line.split(b'\t')[key_idx]
            else:
                try:
                    curr_key = line.split(b'\t')[key_idx]
                except IndexError:
                    curr_key = b''
                if curr_key != qresult_key:
                    break
            qresult_raw += line
        return qresult_raw

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