from Bio.Blast import Record
import xml.sax
from xml.sax.handler import ContentHandler
def _end_hit_descr_item(self):
    """XML v2. Start hit description item."""
    self._descr.append_item(self._hit_descr_item)
    if not self._hit.title:
        self._hit.title = str(self._hit_descr_item)
    self._hit_descr_item = None