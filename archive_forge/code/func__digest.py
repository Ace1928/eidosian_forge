from Bio.Align import MultipleSeqAlignment
from Bio.Seq import Seq
def _digest(self):
    self.dcuts = []
    for enzyme in self.enzymes:
        self._digest_with(enzyme)