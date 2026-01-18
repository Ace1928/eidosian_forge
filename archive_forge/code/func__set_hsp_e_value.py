from Bio.Blast import Record
import xml.sax
from xml.sax.handler import ContentHandler
def _set_hsp_e_value(self):
    """Record the expect value of the HSP (PRIVATE)."""
    self._hsp.expect = float(self._value)
    if self._descr.e is None:
        self._descr.e = float(self._value)