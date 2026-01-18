from Bio.Blast import Record
import xml.sax
from xml.sax.handler import ContentHandler
def _set_hsp_hit_frame(self):
    """Frame of the database sequence if applicable (PRIVATE)."""
    v = int(self._value)
    if len(self._hsp.frame) == 0:
        self._hsp.frame = (0, v)
    else:
        self._hsp.frame += (v,)
    if self._header.application == 'BLASTN':
        self._hsp.strand += ('Plus' if v > 0 else 'Minus',)