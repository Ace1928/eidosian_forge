import textwrap
from collections import defaultdict
from Bio.Align import Alignment
from Bio.Align import interfaces
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
@staticmethod
def _store_per_sequence_annotations(alignment, gs):
    for seqname, annotations in gs.items():
        for record in alignment.sequences:
            if record.id == seqname:
                break
        else:
            raise ValueError(f'Failed to find seqname {seqname}')
        for key, value in annotations.items():
            if key == 'DE':
                record.description = value
            elif key == 'DR':
                record.dbxrefs = value
            else:
                record.annotations[AlignmentIterator.gs_mapping.get(key, key)] = value