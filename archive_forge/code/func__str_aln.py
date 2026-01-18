import warnings
from operator import ge, le
from Bio import BiopythonWarning
from Bio.Align import MultipleSeqAlignment
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.SearchIO._utils import (
from ._base import _BaseHSP
def _str_aln(self):
    lines = []
    aln_span = getattr_str(self, 'aln_span')
    lines.append('  Fragments: 1 (%s columns)' % aln_span)
    if self.query is not None and self.hit is not None:
        try:
            qseq = self.query.seq
        except AttributeError:
            qseq = '?'
        try:
            hseq = self.hit.seq
        except AttributeError:
            hseq = '?'
        simil = ''
        if 'similarity' in self.aln_annotation and isinstance(self.aln_annotation.get('similarity'), str):
            simil = self.aln_annotation['similarity']
        if self.aln_span <= 67:
            lines.append('%10s - %s' % ('Query', qseq))
            if simil:
                lines.append('             %s' % simil)
            lines.append('%10s - %s' % ('Hit', hseq))
        else:
            if self.aln_span - 66 > 3:
                cont = '~' * 3
            else:
                cont = '~' * (self.aln_span - 66)
            lines.append('%10s - %s%s%s' % ('Query', qseq[:59], cont, qseq[-5:]))
            if simil:
                lines.append('             %s%s%s' % (simil[:59], cont, simil[-5:]))
            lines.append('%10s - %s%s%s' % ('Hit', hseq[:59], cont, hseq[-5:]))
    return '\n'.join(lines)