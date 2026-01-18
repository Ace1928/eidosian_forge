from Bio.Blast import Record
import xml.sax
from xml.sax.handler import ContentHandler
def _set_statistics_eff_space(self):
    """Record the effective search space (PRIVATE)."""
    self._blast.effective_search_space = float(self._value)