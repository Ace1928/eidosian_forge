from Bio.Blast import Record
import xml.sax
from xml.sax.handler import ContentHandler
def _set_header_version(self):
    """Version number and date of the BLAST engine (PRIVATE).

        e.g. "BLASTX 2.2.12 [Aug-07-2005]" but there can also be
        variants like "BLASTP 2.2.18+" without the date.

        Save this to put on each blast record object
        """
    parts = self._value.split()
    self._header.version = parts[1]
    if len(parts) >= 3:
        if parts[2][0] == '[' and parts[2][-1] == ']':
            self._header.date = parts[2][1:-1]
        else:
            self._header.date = parts[2]