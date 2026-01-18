from Bio.Blast import Record
import xml.sax
from xml.sax.handler import ContentHandler
def _set_hsp_subject_seq(self):
    """Record the alignment string for the database (PRIVATE)."""
    self._hsp.sbjct = self._value