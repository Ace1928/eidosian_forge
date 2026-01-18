import textwrap
from collections import defaultdict
from Bio.Align import Alignment
from Bio.Align import interfaces
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
@staticmethod
def _store_per_sequence_and_per_column_annotations(alignment, gr):
    for seqname, letter_annotations in gr.items():
        for record in alignment.sequences:
            if record.id == seqname:
                break
        else:
            raise ValueError(f'Failed to find seqname {seqname}')
        for key, letter_annotation in letter_annotations.items():
            feature = AlignmentIterator.gr_mapping.get(key, key)
            if key == 'CSA':
                letter_annotation = letter_annotation.replace('-', '')
            else:
                letter_annotation = letter_annotation.replace('.', '')
            record.letter_annotations[feature] = letter_annotation