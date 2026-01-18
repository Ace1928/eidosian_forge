from Bio.Blast import Record
import xml.sax
from xml.sax.handler import ContentHandler
def _set_hsp_query_end(self):
    """Offset of query at the end of the alignment (one-offset) (PRIVATE)."""
    self._hsp.query_end = int(self._value)