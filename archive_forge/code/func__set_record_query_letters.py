from Bio.Blast import Record
import xml.sax
from xml.sax.handler import ContentHandler
def _set_record_query_letters(self):
    """Record the length of the query (PRIVATE)."""
    self._blast.query_letters = int(self._value)