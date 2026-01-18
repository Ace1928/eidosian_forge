from Bio.Blast import Record
import xml.sax
from xml.sax.handler import ContentHandler
def _end_hsp(self):
    if self._hsp.frame and len(self._hsp.frame) == 1:
        self._hsp.frame += (0,)