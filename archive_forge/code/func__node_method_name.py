from Bio.Blast import Record
import xml.sax
from xml.sax.handler import ContentHandler
def _node_method_name(self, name):
    if self._method_name_level == 1:
        return name
    return '/'.join(self._tag[-self._method_name_level:])