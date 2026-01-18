from Bio.Seq import Seq
from Bio.Seq import SequenceDataAbstractBaseClass
from Bio.SeqRecord import SeqRecord
from . import _twoBitIO  # type: ignore
from .Interfaces import SequenceIterator
class TwoBitIterator(SequenceIterator):
    """Parser for UCSC twoBit (.2bit) files."""

    def __init__(self, source):
        """Read the file index."""
        super().__init__(source, mode='b', fmt='twoBit')
        self.should_close_stream = False
        stream = self.stream
        data = stream.read(4)
        if not data:
            raise ValueError('Empty file.')
        byteorders = ('little', 'big')
        dtypes = ('<u4', '>u4')
        for byteorder, dtype in zip(byteorders, dtypes):
            signature = int.from_bytes(data, byteorder)
            if signature == 440477507:
                break
        else:
            raise ValueError('Unknown signature')
        self.byteorder = byteorder
        data = stream.read(4)
        version = int.from_bytes(data, byteorder, signed=False)
        if version == 1:
            raise ValueError('version-1 twoBit files with 64-bit offsets for index are currently not supported')
        if version != 0:
            raise ValueError('Found unexpected file version %u; aborting' % version)
        data = stream.read(4)
        sequenceCount = int.from_bytes(data, byteorder, signed=False)
        data = stream.read(4)
        reserved = int.from_bytes(data, byteorder, signed=False)
        if reserved != 0:
            raise ValueError('Found non-zero reserved field; aborting')
        sequences = {}
        for i in range(sequenceCount):
            data = stream.read(1)
            nameSize = int.from_bytes(data, byteorder, signed=False)
            data = stream.read(nameSize)
            name = data.decode('ASCII')
            data = stream.read(4)
            offset = int.from_bytes(data, byteorder, signed=False)
            sequences[name] = (stream, offset)
        self.sequences = sequences
        for name, (stream, offset) in sequences.items():
            stream.seek(offset)
            data = stream.read(4)
            dnaSize = int.from_bytes(data, byteorder, signed=False)
            sequence = _TwoBitSequenceData(stream, offset, dnaSize)
            data = stream.read(4)
            nBlockCount = int.from_bytes(data, byteorder, signed=False)
            nBlockStarts = np.fromfile(stream, dtype=dtype, count=nBlockCount)
            nBlockSizes = np.fromfile(stream, dtype=dtype, count=nBlockCount)
            sequence.nBlocks = np.empty((nBlockCount, 2), dtype='uint32')
            sequence.nBlocks[:, 0] = nBlockStarts
            sequence.nBlocks[:, 1] = nBlockStarts + nBlockSizes
            data = stream.read(4)
            maskBlockCount = int.from_bytes(data, byteorder, signed=False)
            maskBlockStarts = np.fromfile(stream, dtype=dtype, count=maskBlockCount)
            maskBlockSizes = np.fromfile(stream, dtype=dtype, count=maskBlockCount)
            sequence.maskBlocks = np.empty((maskBlockCount, 2), dtype='uint32')
            sequence.maskBlocks[:, 0] = maskBlockStarts
            sequence.maskBlocks[:, 1] = maskBlockStarts + maskBlockSizes
            data = stream.read(4)
            reserved = int.from_bytes(data, byteorder, signed=False)
            if reserved != 0:
                raise ValueError('Found non-zero reserved field %u' % reserved)
            sequence.offset = stream.tell()
            sequences[name] = sequence

    def parse(self, stream):
        """Iterate over the sequences in the file."""
        for name, sequence in self.sequences.items():
            sequence = Seq(sequence)
            record = SeqRecord(sequence, id=name)
            yield record

    def __getitem__(self, name):
        """Return sequence associated with given name as a SeqRecord object."""
        try:
            sequence = self.sequences[name]
        except ValueError:
            raise KeyError(name) from None
        sequence = Seq(sequence)
        return SeqRecord(sequence, id=name)

    def keys(self):
        """Return a list with the names of the sequences in the file."""
        return self.sequences.keys()

    def __len__(self):
        """Return number of sequences."""
        return len(self.sequences)