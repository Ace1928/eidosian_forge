from Bio.Blast import Record
import xml.sax
from xml.sax.handler import ContentHandler
def _set_hsp_hit_from(self):
    """Offset of the database at the start of the alignment (one-offset) (PRIVATE)."""
    self._hsp.sbjct_start = int(self._value)