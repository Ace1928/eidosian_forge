from Bio.Blast import Record
import xml.sax
from xml.sax.handler import ContentHandler
def _set_statistics_entropy(self):
    """Karlin-Altschul parameter H (PRIVATE)."""
    self._blast.ka_params = self._blast.ka_params + (float(self._value),)