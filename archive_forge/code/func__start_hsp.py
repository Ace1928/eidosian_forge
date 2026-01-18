from Bio.Blast import Record
import xml.sax
from xml.sax.handler import ContentHandler
def _start_hsp(self):
    self._hsp = Record.HSP()
    self._hsp.positives = None
    self._hit.hsps.append(self._hsp)
    self._descr.num_alignments += 1
    self._blast.multiple_alignment.append(Record.MultipleAlignment())
    self._mult_al = self._blast.multiple_alignment[-1]