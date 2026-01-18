from Bio.Blast import Record
import xml.sax
from xml.sax.handler import ContentHandler
def _set_hsp_identity(self):
    """Record the number of identities in the alignment (PRIVATE)."""
    v = int(self._value)
    self._hsp.identities = v
    if self._hsp.positives is None:
        self._hsp.positives = v