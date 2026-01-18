from Bio.Blast import Record
import xml.sax
from xml.sax.handler import ContentHandler
def _set_parameters_gap_extend(self):
    """Gap extension cose (-E) (PRIVATE)."""
    self._parameters.gap_penalties = (self._parameters.gap_penalties, int(self._value))