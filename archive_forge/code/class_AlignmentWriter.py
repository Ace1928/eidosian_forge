from Bio.Align import Alignment
from Bio.Align import interfaces
from Bio.Seq import Seq, reverse_complement
from Bio.SeqRecord import SeqRecord
class AlignmentWriter(interfaces.AlignmentWriter):
    """Mauve xmfa alignment writer."""
    fmt = 'Mauve'

    def __init__(self, target, metadata=None, identifiers=None):
        """Create an AlignmentWriter object.

        Arguments:
         - target       - output stream or file name
         - metadata     - metadata to be included in the output. If metadata
                          is None, then the alignments object to be written
                          must have an attribute `metadata`.
         - identifiers  - list of the IDs of the sequences included in the
                          alignment. Sequences will be numbered according to
                          their index in this list. If identifiers is None,
                          then the alignments object to be written must have
                          an attribute `identifiers`.
        """
        super().__init__(target)
        self._metadata = metadata
        self._identifiers = identifiers

    def write_header(self, stream, alignments):
        """Write the file header to the output file."""
        metadata = self._metadata
        format_version = metadata.get('FormatVersion', 'Mauve1')
        line = f'#FormatVersion {format_version}\n'
        stream.write(line)
        identifiers = self._identifiers
        filename = metadata.get('File')
        if filename is None:
            for index, filename in enumerate(identifiers):
                number = index + 1
                line = f'#Sequence{number}File\t{filename}\n'
                stream.write(line)
                line = f'#Sequence{number}Format\tFastA\n'
                stream.write(line)
        else:
            for number, identifier in enumerate(identifiers):
                assert number == int(identifier)
                number += 1
                line = f'#Sequence{number}File\t{filename}\n'
                stream.write(line)
                line = f'#Sequence{number}Entry\t{number}\n'
                stream.write(line)
                line = f'#Sequence{number}Format\tFastA\n'
                stream.write(line)
        backbone_file = metadata.get('BackboneFile')
        if backbone_file is not None:
            line = f'#BackboneFile\t{backbone_file}\n'
            stream.write(line)

    def write_file(self, stream, alignments):
        """Write a file with the alignments, and return the number of alignments.

        alignments - A Bio.Align.mauve.AlignmentIterator object.
        """
        metadata = self._metadata
        if metadata is None:
            try:
                metadata = alignments.metadata
            except AttributeError:
                raise ValueError('alignments do not have an attribute `metadata`')
            else:
                self._metadata = metadata
        identifiers = self._identifiers
        if identifiers is None:
            try:
                identifiers = alignments.identifiers
            except AttributeError:
                raise ValueError('alignments do not have an attribute `identifiers`')
            else:
                self._identifiers = identifiers
        count = interfaces.AlignmentWriter.write_file(self, stream, alignments)
        return count

    def format_alignment(self, alignment):
        """Return a string with a single alignment in the Mauve format."""
        metadata = self._metadata
        n, m = alignment.shape
        if n == 0:
            raise ValueError('Must have at least one sequence')
        if m == 0:
            raise ValueError('Non-empty sequences are required')
        filename = metadata.get('File')
        lines = []
        for i in range(n):
            identifier = alignment.sequences[i].id
            start = alignment.coordinates[i, 0]
            end = alignment.coordinates[i, -1]
            if start <= end:
                strand = '+'
            else:
                strand = '-'
                start, end = (end, start)
            if start == end:
                assert start == 0
            else:
                start += 1
            sequence = alignment[i]
            if filename is None:
                number = self._identifiers.index(identifier) + 1
                line = f'> {number}:{start}-{end} {strand} {identifier}\n'
            else:
                number = int(identifier) + 1
                line = f'> {number}:{start}-{end} {strand} {filename}\n'
            lines.append(line)
            line = f'{sequence}\n'
            lines.append(line)
        lines.append('=\n')
        return ''.join(lines)