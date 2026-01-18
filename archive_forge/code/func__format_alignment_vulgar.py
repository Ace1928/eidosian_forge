import numpy as np
from Bio.Align import Alignment
from Bio.Align import interfaces
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
def _format_alignment_vulgar(self, alignment):
    """Return a string with a single alignment formatted as one vulgar line."""
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
    words = ['vulgar:', query_id, str(query_start), str(query_end), query_strand, target_id, str(target_start), str(target_end), target_strand, str(score)]
    try:
        operations = alignment.operations
    except AttributeError:
        for step in steps.transpose():
            target_step, query_step = step
            if target_step == query_step:
                operation = 'M'
            elif query_step == 0:
                operation = 'G'
            elif target_step == 0:
                operation = 'G'
            elif query_molecule_type == 'protein' and target_molecule_type != 'protein':
                operation = 'M'
            elif query_molecule_type != 'protein' and target_molecule_type == 'protein':
                operation = 'M'
            else:
                raise ValueError('Both target and query step are zero')
            words.append(operation)
            words.append(str(query_step))
            words.append(str(target_step))
    else:
        steps = steps.transpose()
        operations = operations.decode()
        n = len(operations)
        i = 0
        while i < n:
            target_step, query_step = steps[i]
            operation = operations[i]
            if operation == 'M':
                if target_step == query_step:
                    pass
                elif target_step == 3 * query_step:
                    assert query_molecule_type == 'protein'
                    assert target_molecule_type != 'protein'
                elif query_step == 3 * target_step:
                    assert query_molecule_type != 'protein'
                    assert target_molecule_type == 'protein'
                else:
                    raise ValueError("Unexpected steps target %d, query %d for operation 'M'" % (target_step, query_step))
            elif operation == '5':
                assert target_step == 2 or query_step == 2
            elif operation == 'N':
                operation = 'I'
                assert query_step == 0 or target_step == 0
            elif operation == '3':
                assert target_step == 2 or query_step == 2
            elif operation == 'C':
                assert target_step == query_step
            elif operation == 'D':
                assert query_step == 0
                operation = 'G'
            elif operation == 'I':
                assert target_step == 0
                operation = 'G'
            elif operation == 'U':
                if target_step == 0:
                    assert query_step > 0
                    i += 1
                    target_step, dummy = steps[i]
                    assert dummy == 0
                if query_step == 0:
                    assert target_step > 0
                    i += 1
                    dummy, query_step = steps[i]
                    assert dummy == 0
                operation = operations[i]
                assert operation == 'U'
                operation = 'N'
            elif operation == 'S':
                step = target_step
            elif operation == 'F':
                step = target_step
            else:
                raise ValueError('Unknown operation %s' % operation)
            words.append(operation)
            words.append(str(query_step))
            words.append(str(target_step))
            i += 1
    line = ' '.join(words) + '\n'
    return line