import re
from typing import List
from Bio.Align import MultipleSeqAlignment
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from .Interfaces import AlignmentIterator
from .Interfaces import SequentialAlignmentWriter
class MauveIterator(AlignmentIterator):
    """Mauve xmfa alignment iterator."""
    _ids: List[str] = []

    def __next__(self):
        """Parse the next alignment from the handle."""
        handle = self.handle
        line = handle.readline()
        if not line:
            raise StopIteration
        while line and line.strip().startswith('#'):
            line = handle.readline()
        seqs = {}
        seq_regions = {}
        passed_end_alignment = False
        latest_id = None
        while True:
            if not line:
                break
            line = line.strip()
            if line.startswith('='):
                break
            elif line.startswith('>'):
                m = XMFA_HEADER_REGEX_BIOPYTHON.match(line)
                if not m:
                    m = XMFA_HEADER_REGEX.match(line)
                    if not m:
                        raise ValueError('Malformed header line: %s', line)
                parsed_id = m.group('id')
                parsed_data = {}
                for key in ('start', 'end', 'id', 'strand', 'name', 'realname'):
                    try:
                        value = m.group(key)
                        if key == 'start':
                            value = int(value)
                            if value > 0:
                                value -= 1
                        if key == 'end':
                            value = int(value)
                        parsed_data[key] = value
                    except IndexError:
                        pass
                seq_regions[parsed_id] = parsed_data
                if parsed_id not in self._ids:
                    self._ids.append(parsed_id)
                seqs.setdefault(parsed_id, '')
                latest_id = parsed_id
            else:
                assert not passed_end_alignment
                if latest_id is None:
                    raise ValueError('Saw sequence before definition line')
                seqs[latest_id] += line
            line = handle.readline()
        assert len(seqs) <= len(self._ids)
        self.ids = self._ids
        self.sequences = seqs
        if self._ids and seqs:
            alignment_length = max(map(len, list(seqs.values())))
            records = []
            for id in self._ids:
                if id not in seqs or len(seqs[id]) == 0 or len(seqs[id]) == 0:
                    seq = '-' * alignment_length
                else:
                    seq = seqs[id]
                if alignment_length != len(seq):
                    raise ValueError('Sequences have different lengths, or repeated identifier')
                if id not in seq_regions:
                    continue
                if seq_regions[id]['start'] != 0 or seq_regions[id]['end'] != 0:
                    suffix = '/{start}-{end}'.format(**seq_regions[id])
                    if 'realname' in seq_regions[id]:
                        corrected_id = seq_regions[id]['realname']
                    else:
                        corrected_id = seq_regions[id]['name']
                    if corrected_id.count(suffix) == 0:
                        corrected_id += suffix
                elif 'realname' in seq_regions[id]:
                    corrected_id = seq_regions[id]['realname']
                else:
                    corrected_id = seq_regions[id]['name']
                record = SeqRecord(Seq(seq), id=corrected_id, name=id)
                record.annotations['start'] = seq_regions[id]['start']
                record.annotations['end'] = seq_regions[id]['end']
                record.annotations['strand'] = 1 if seq_regions[id]['strand'] == '+' else -1
                records.append(record)
            return MultipleSeqAlignment(records)
        else:
            raise StopIteration