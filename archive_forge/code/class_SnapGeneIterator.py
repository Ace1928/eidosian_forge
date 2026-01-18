from GSL Biotech LLC.
from datetime import datetime
from re import sub
from struct import unpack
from xml.dom.minidom import parseString
from Bio.Seq import Seq
from Bio.SeqFeature import SimpleLocation
from Bio.SeqFeature import SeqFeature
from Bio.SeqRecord import SeqRecord
from .Interfaces import SequenceIterator
class SnapGeneIterator(SequenceIterator):
    """Parser for SnapGene files."""

    def __init__(self, source):
        """Parse a SnapGene file and return a SeqRecord object.

        Argument source is a file-like object or a path to a file.

        Note that a SnapGene file can only contain one sequence, so this
        iterator will always return a single record.
        """
        super().__init__(source, mode='b', fmt='SnapGene')

    def parse(self, handle):
        """Start parsing the file, and return a SeqRecord generator."""
        records = self.iterate(handle)
        return records

    def iterate(self, handle):
        """Iterate over the records in the SnapGene file."""
        record = SeqRecord(None)
        packets = _iterate(handle)
        try:
            packet_type, length, data = next(packets)
        except StopIteration:
            raise ValueError('Empty file.') from None
        if packet_type != 9:
            raise ValueError('The file does not start with a SnapGene cookie packet')
        _parse_cookie_packet(length, data, record)
        for packet_type, length, data in packets:
            handler = _packet_handlers.get(packet_type)
            if handler is not None:
                handler(length, data, record)
        if not record.seq:
            raise ValueError('No DNA packet in file')
        yield record