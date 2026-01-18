from Bio.Blast import Record
import xml.sax
from xml.sax.handler import ContentHandler
def _set_hsp_query_frame(self):
    """Frame of the query if applicable (PRIVATE)."""
    v = int(self._value)
    self._hsp.frame = (v,)
    if self._header.application == 'BLASTN':
        self._hsp.strand = ('Plus' if v > 0 else 'Minus',)