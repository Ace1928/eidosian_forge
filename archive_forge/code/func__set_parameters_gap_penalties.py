from Bio.Blast import Record
import xml.sax
from xml.sax.handler import ContentHandler
def _set_parameters_gap_penalties(self):
    """Gap existence cost (-G) (PRIVATE)."""
    self._parameters.gap_penalties = int(self._value)