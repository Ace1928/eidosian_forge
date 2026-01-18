import warnings
from re import match
from struct import pack
from struct import unpack
from Bio import BiopythonWarning
from Bio.Seq import Seq
from Bio.SeqFeature import ExactPosition
from Bio.SeqFeature import SimpleLocation
from Bio.SeqFeature import SeqFeature
from Bio.SeqRecord import SeqRecord
from .Interfaces import SequenceIterator
from .Interfaces import SequenceWriter
class XdnaIterator(SequenceIterator):
    """Parser for Xdna files."""

    def __init__(self, source):
        """Parse a Xdna file and return a SeqRecord object.

        Argument source is a file-like object in binary mode or a path to a file.

        Note that this is an "iterator" in name only since an Xdna file always
        contain a single sequence.

        """
        super().__init__(source, mode='b', fmt='Xdna')

    def parse(self, handle):
        """Start parsing the file, and return a SeqRecord generator."""
        header = handle.read(112)
        if not header:
            raise ValueError('Empty file.')
        if len(header) < 112:
            raise ValueError('Improper header, cannot read 112 bytes from handle')
        records = self.iterate(handle, header)
        return records

    def iterate(self, handle, header):
        """Parse the file and generate SeqRecord objects."""
        version, seq_type, topology, length, neg_length, com_length = unpack('>BBB25xII60xI12x', header)
        if version != 0:
            raise ValueError('Unsupported XDNA version')
        if seq_type not in _seq_types:
            raise ValueError('Unknown sequence type')
        sequence = _read(handle, length).decode('ASCII')
        comment = _read(handle, com_length).decode('ASCII')
        name = comment.split(' ')[0]
        record = SeqRecord(Seq(sequence), description=comment, name=name, id=name)
        if _seq_types[seq_type]:
            record.annotations['molecule_type'] = _seq_types[seq_type]
        if topology in _seq_topologies:
            record.annotations['topology'] = _seq_topologies[topology]
        if len(handle.read(1)) == 1:
            _read_overhang(handle)
            _read_overhang(handle)
            num_features = unpack('>B', _read(handle, 1))[0]
            while num_features > 0:
                _read_feature(handle, record)
                num_features -= 1
        yield record