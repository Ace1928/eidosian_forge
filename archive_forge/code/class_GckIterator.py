from Textco BioSoftware, Inc.
from struct import unpack
from Bio.Seq import Seq
from Bio.SeqFeature import SimpleLocation
from Bio.SeqFeature import SeqFeature
from Bio.SeqRecord import SeqRecord
from .Interfaces import SequenceIterator
class GckIterator(SequenceIterator):
    """Parser for GCK files."""

    def __init__(self, source):
        """Break up a GCK file into SeqRecord objects."""
        super().__init__(source, mode='b', fmt='GCK')

    def parse(self, handle):
        """Start parsing the file, and return a SeqRecord generator.

        Note that a GCK file can only contain one sequence, so this
        iterator will always return a single record.
        """
        records = _parse(handle)
        return records