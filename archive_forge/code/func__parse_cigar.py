import numpy as np
from Bio.Align import Alignment
from Bio.Align import interfaces
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
@staticmethod
def _parse_cigar(words):
    query_id = words[0]
    query_start = int(words[1])
    query_end = int(words[2])
    query_strand = words[3]
    target_id = words[4]
    target_start = int(words[5])
    target_end = int(words[6])
    target_strand = words[7]
    score = float(words[8])
    qs = 0
    ts = 0
    n = (len(words) - 8) // 2
    coordinates = np.empty((2, n + 1), int)
    coordinates[0, 0] = ts
    coordinates[1, 0] = qs
    for i, (operation, step) in enumerate(zip(words[9::2], words[10::2])):
        step = int(step)
        if operation == 'M':
            ts += step
            qs += step
        elif operation == 'I':
            if query_strand == '.' and target_strand != '.':
                qs += step * 3
            else:
                qs += step
        elif operation == 'D':
            if target_strand == '.' and query_strand != '.':
                ts += step * 3
            else:
                ts += step
        else:
            raise ValueError('Unknown operation %s in cigar string' % operation)
        coordinates[0, i + 1] = ts
        coordinates[1, i + 1] = qs
    if target_strand == '+':
        coordinates[0, :] += target_start
        target_length = target_end
        target_molecule_type = None
    elif target_strand == '-':
        coordinates[0, :] = target_start - coordinates[0, :]
        target_length = target_start
        target_molecule_type = None
    elif target_strand == '.':
        if query_strand != '.':
            coordinates[0, :] = (coordinates[0, :] + 2) // 3
        coordinates[0, :] += target_start
        target_molecule_type = 'protein'
        target_length = target_end
    if query_strand == '+':
        coordinates[1, :] += query_start
        query_length = query_end
        query_molecule_type = None
    elif query_strand == '-':
        coordinates[1, :] = query_start - coordinates[1, :]
        query_length = query_start
        query_molecule_type = None
    elif query_strand == '.':
        if target_strand != '.':
            coordinates[1, :] = -(coordinates[1, :] // -3)
        coordinates[1, :] += query_start
        query_molecule_type = 'protein'
        query_length = query_end
    target_seq = Seq(None, length=target_length)
    query_seq = Seq(None, length=query_length)
    target = SeqRecord(target_seq, id=target_id, description='')
    query = SeqRecord(query_seq, id=query_id, description='')
    if target_molecule_type is not None:
        target.annotations['molecule_type'] = target_molecule_type
    if query_molecule_type is not None:
        query.annotations['molecule_type'] = query_molecule_type
    alignment = Alignment([target, query], coordinates)
    alignment.score = score
    return alignment