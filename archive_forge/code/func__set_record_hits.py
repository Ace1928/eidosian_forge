from Bio.Blast import Record
import xml.sax
from xml.sax.handler import ContentHandler
def _set_record_hits(self):
    """Hits to the database sequences, one for every sequence (PRIVATE)."""
    self._blast.num_hits = int(self._value)