from Bio.Blast import Record
import xml.sax
from xml.sax.handler import ContentHandler
def _set_hsp_gaps(self):
    """Record the number of gaps in the alignment (PRIVATE)."""
    self._hsp.gaps = int(self._value)