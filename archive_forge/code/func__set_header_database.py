from Bio.Blast import Record
import xml.sax
from xml.sax.handler import ContentHandler
def _set_header_database(self):
    """Record the database(s) searched (PRIVATE).

        Save this to put on each blast record object
        """
    self._header.database = self._value