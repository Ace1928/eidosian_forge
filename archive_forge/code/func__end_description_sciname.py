from Bio.Blast import Record
import xml.sax
from xml.sax.handler import ContentHandler
def _end_description_sciname(self):
    self._hit_descr_item.sciname = self._value