from Bio.Blast import Record
import xml.sax
from xml.sax.handler import ContentHandler
def _set_statistics_hsp_len(self):
    """Record the effective HSP length (PRIVATE)."""
    self._blast.effective_hsp_length = int(self._value)