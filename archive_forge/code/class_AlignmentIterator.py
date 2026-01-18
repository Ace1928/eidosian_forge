from Bio.Align import Alignment
from Bio.Align import interfaces
from Bio.Seq import Seq, reverse_complement
from Bio.SeqRecord import SeqRecord
class AlignmentIterator(interfaces.AlignmentIterator):
    """Mauve xmfa alignment iterator."""
    fmt = 'Mauve'

    def _read_header(self, stream):
        metadata = {}
        prefix = 'Sequence'
        suffixes = ('File', 'Entry', 'Format')
        id_info = {}
        for suffix in suffixes:
            id_info[suffix] = []
        for line in stream:
            if not line.startswith('#'):
                self._line = line.strip()
                break
            key, value = line[1:].split()
            if key.startswith(prefix):
                for suffix in suffixes:
                    if key.endswith(suffix):
                        break
                else:
                    raise ValueError("Unexpected keyword '%s'" % key)
                if suffix == 'Entry':
                    value = int(value) - 1
                seq_num = int(key[len(prefix):-len(suffix)])
                id_info[suffix].append(value)
                assert seq_num == len(id_info[suffix])
            else:
                metadata[key] = value.strip()
        else:
            if not metadata:
                raise ValueError('Empty file.') from None
        if len(set(id_info['File'])) == 1:
            metadata['File'] = id_info['File'][0]
            self.identifiers = [str(entry) for entry in id_info['Entry']]
        else:
            assert len(set(id_info['File'])) == len(id_info['File'])
            self.identifiers = id_info['File']
        self.metadata = metadata

    def _parse_description(self, line):
        assert line.startswith('>')
        locus, strand, comments = line[1:].split(None, 2)
        seq_num, start_end = locus.split(':')
        seq_num = int(seq_num) - 1
        identifier = self.identifiers[seq_num]
        assert strand in '+-'
        start, end = start_end.split('-')
        start = int(start)
        end = int(end)
        if start == 0:
            assert end == 0
        else:
            start -= 1
        return (identifier, start, end, strand, comments)

    def _read_next_alignment(self, stream):
        descriptions = []
        seqs = []
        try:
            line = self._line
        except AttributeError:
            pass
        else:
            del self._line
            description = self._parse_description(line)
            identifier, start, end, strand, comments = description
            descriptions.append(description)
            seqs.append('')
        for line in stream:
            line = line.strip()
            if line.startswith('='):
                coordinates = Alignment.infer_coordinates(seqs)
                records = []
                for index, (description, seq) in enumerate(zip(descriptions, seqs)):
                    identifier, start, end, strand, comments = description
                    seq = seq.replace('-', '')
                    assert len(seq) == end - start
                    if strand == '+':
                        pass
                    elif strand == '-':
                        seq = reverse_complement(seq)
                        coordinates[index, :] = len(seq) - coordinates[index, :]
                    else:
                        raise ValueError("Unexpected strand '%s'" % strand)
                    coordinates[index] += start
                    if start == 0:
                        seq = Seq(seq)
                    else:
                        seq = Seq({start: seq}, length=end)
                    record = SeqRecord(seq, id=identifier, description=comments)
                    records.append(record)
                return Alignment(records, coordinates)
            elif line.startswith('>'):
                description = self._parse_description(line)
                identifier, start, end, strand, comments = description
                descriptions.append(description)
                seqs.append('')
            else:
                seqs[-1] += line