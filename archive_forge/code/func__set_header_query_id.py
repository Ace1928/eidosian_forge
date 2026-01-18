from Bio.Blast import Record
import xml.sax
from xml.sax.handler import ContentHandler
def _set_header_query_id(self):
    """Record the identifier of the query (PRIVATE).

        Important in old pre 2.2.14 BLAST, for recent versions
        <Iteration_query-ID> is enough
        """
    self._header.query_id = self._value