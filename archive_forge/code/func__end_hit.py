from Bio.Blast import Record
import xml.sax
from xml.sax.handler import ContentHandler
def _end_hit(self):
    """Clear variables (PRIVATE)."""
    self._blast.multiple_alignment = None
    self._hit = None
    self._descr = None