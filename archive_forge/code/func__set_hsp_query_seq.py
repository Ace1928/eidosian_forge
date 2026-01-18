from Bio.Blast import Record
import xml.sax
from xml.sax.handler import ContentHandler
def _set_hsp_query_seq(self):
    """Record the alignment string for the query (PRIVATE)."""
    self._hsp.query = self._value