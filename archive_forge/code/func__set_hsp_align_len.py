from Bio.Blast import Record
import xml.sax
from xml.sax.handler import ContentHandler
def _set_hsp_align_len(self):
    """Record the length of the alignment (PRIVATE)."""
    self._hsp.align_length = int(self._value)