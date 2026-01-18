import re
from ._base import _BaseExonerateParser, _STRAND_MAP
from .exonerate_vulgar import ExonerateVulgarIndexer
class ExonerateCigarIndexer(ExonerateVulgarIndexer):
    """Indexer class for exonerate cigar lines."""
    _parser = ExonerateCigarParser
    _query_mark = b'cigar'

    def get_qresult_id(self, pos):
        """Return the query ID of the nearest cigar line."""
        handle = self._handle
        handle.seek(pos)
        line = handle.readline()
        assert line.startswith(self._query_mark), line
        id = re.search(_RE_CIGAR, line.decode())
        return id.group(1)