from Bio.Blast import Record
import xml.sax
from xml.sax.handler import ContentHandler
def _start_hit_descr_item(self):
    """XML v2. Start hit description item."""
    self._hit_descr_item = Record.DescriptionExtItem()