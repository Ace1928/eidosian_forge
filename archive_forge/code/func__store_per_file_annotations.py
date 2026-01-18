import textwrap
from collections import defaultdict
from Bio.Align import Alignment
from Bio.Align import interfaces
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
@staticmethod
def _store_per_file_annotations(alignment, gf, rows):
    for key, value in gf.items():
        if key == 'WK':
            lines = iter(value)
            references = []
            for line in lines:
                reference = ''
                while line.endswith('/'):
                    reference += line[:-1]
                    line = next(lines)
                reference += line
                references.append(reference)
            value = references
        elif key in ('SM', 'CC', '**'):
            value = ' '.join(value)
        elif key == 'SQ':
            assert len(value) == 1
            if int(value.pop()) != rows:
                raise ValueError('Inconsistent number of sequences in alignment')
            continue
        elif key == 'AU':
            pass
        else:
            assert len(value) == 1, (key, value)
            value = value.pop()
        try:
            alignment.annotations[AlignmentIterator.gf_mapping[key]] = value
        except KeyError:
            pass