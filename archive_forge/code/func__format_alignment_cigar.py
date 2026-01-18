import numpy as np
from Bio.Align import Alignment
from Bio.Align import interfaces
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
def _format_alignment_cigar(self, alignment):
    """Return a string with a single alignment formatted as a cigar line."""
    if not isinstance(alignment, Alignment):
        raise TypeError('Expected an Alignment object')
    coordinates = alignment.coordinates
    target_start = coordinates[0, 0]
    target_end = coordinates[0, -1]
    query_start = coordinates[1, 0]
    query_end = coordinates[1, -1]
    steps = np.diff(coordinates)
    query = alignment.query
    target = alignment.target
    try:
        query_id = query.id
    except AttributeError:
        query_id = 'query'
    try:
        target_id = target.id
    except AttributeError:
        target_id = 'target'
    try:
        target_molecule_type = target.annotations['molecule_type']
    except (AttributeError, KeyError):
        target_molecule_type = None
    if target_molecule_type == 'protein':
        target_strand = '.'
    elif target_start <= target_end:
        target_strand = '+'
    elif target_start > target_end:
        target_strand = '-'
        steps[0, :] = -steps[0, :]
    try:
        query_molecule_type = query.annotations['molecule_type']
    except (AttributeError, KeyError):
        query_molecule_type = None
    if query_molecule_type == 'protein':
        query_strand = '.'
    elif query_start <= query_end:
        query_strand = '+'
    elif query_start > query_end:
        query_strand = '-'
        steps[1, :] = -steps[1, :]
    score = format(alignment.score, 'g')
    words = ['cigar:', query_id, str(query_start), str(query_end), query_strand, target_id, str(target_start), str(target_end), target_strand, score]
    try:
        operations = alignment.operations
    except AttributeError:
        for step in steps.transpose():
            target_step, query_step = step
            if target_step == query_step:
                operation = 'M'
                step = target_step
            elif query_step == 0:
                operation = 'D'
                step = target_step
            elif target_step == 0:
                operation = 'I'
                step = query_step
            elif target_molecule_type != 'protein' and query_molecule_type == 'protein':
                operation = 'M'
                step = target_step
            elif target_molecule_type == 'protein' and query_molecule_type != 'protein':
                operation = 'M'
                step = query_step
            else:
                raise ValueError('Unexpected step target %d, query %d for molecule type %s, %s' % (target_step, query_step, target_molecule_type, query_molecule_type))
            words.append(operation)
            words.append(str(step))
    else:
        for step, operation in zip(steps.transpose(), operations.decode()):
            target_step, query_step = step
            if operation == 'M':
                if target_step == query_step:
                    step = target_step
                elif target_step == 3 * query_step:
                    step = target_step
                    assert query_molecule_type == 'protein'
                    assert target_molecule_type != 'protein'
                elif query_step == 3 * target_step:
                    step = query_step
                    assert query_molecule_type != 'protein'
                    assert target_molecule_type == 'protein'
                else:
                    raise ValueError("Unexpected steps target %d, query %s for operation 'M'")
            elif operation == '5':
                if query_step == 0:
                    step = target_step
                    operation = 'D'
                elif target_step == 0:
                    step = query_step
                    operation = 'I'
                else:
                    assert query_step == target_step
                    step = target_step
                    operation = 'M'
            elif operation == 'N':
                if query_step == 0:
                    step = target_step
                    operation = 'D'
                elif target_step == 0:
                    step = query_step
                    operation = 'I'
                else:
                    raise ValueError('Unexpected intron with steps target %d, query %d' % (target_step, query_step))
            elif operation == '3':
                if query_step == 0:
                    step = target_step
                    operation = 'D'
                elif target_step == 0:
                    step = query_step
                    operation = 'I'
                else:
                    assert query_step == target_step
                    step = target_step
                    operation = 'M'
            elif operation == 'C':
                assert target_step == query_step
                step = target_step
                operation = 'M'
            elif operation == 'D':
                assert query_step == 0
                step = target_step
                operation = 'D'
            elif operation == 'I':
                assert target_step == 0
                step = query_step
            elif operation == 'U':
                if target_step > 0:
                    operation = 'D'
                    words.append(operation)
                    words.append(str(target_step))
                if query_step > 0:
                    operation = 'I'
                    words.append(operation)
                    words.append(str(query_step))
                continue
            elif operation == 'S':
                if target_step > 0:
                    operation = 'D'
                    words.append(operation)
                    words.append(str(target_step))
                if query_step > 0:
                    operation = 'I'
                    words.append(operation)
                    words.append(str(query_step))
                continue
            elif operation == 'F':
                if target_step == 0:
                    step = query_step
                    operation = 'I'
                elif query_step == 0:
                    step = target_step
                    operation = 'D'
                else:
                    raise ValueError('Expected target step or query step to be 0')
            else:
                raise ValueError('Unknown operation %s' % operation)
            words.append(operation)
            words.append(str(step))
    line = ' '.join(words) + '\n'
    return line