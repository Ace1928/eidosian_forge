from Bio.Blast import Record
import xml.sax
from xml.sax.handler import ContentHandler
def _set_record_query_id(self):
    """Record the identifier of the query (PRIVATE)."""
    self._blast.query_id = self._value