from Bio.Blast import Record
import xml.sax
from xml.sax.handler import ContentHandler
def _set_hsp_bit_score(self):
    """Record the Bit score of HSP (PRIVATE)."""
    self._hsp.bits = float(self._value)
    if self._descr.bits is None:
        self._descr.bits = float(self._value)