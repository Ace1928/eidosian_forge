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
class XdnaWriter(SequenceWriter):
    """Write files in the Xdna format."""

    def __init__(self, target):
        """Initialize an Xdna writer object.

        Arguments:
         - target - Output stream opened in binary mode, or a path to a file.

        """
        super().__init__(target, mode='wb')

    def write_file(self, records):
        """Write the specified record to a Xdna file.

        Note that the function expects a list (or iterable) of records
        as per the SequenceWriter interface, but the list should contain
        only one record as the Xdna format is a mono-record format.
        """
        records = iter(records)
        try:
            record = next(records)
        except StopIteration:
            raise ValueError('Must have one sequence') from None
        try:
            next(records)
            raise ValueError('More than one sequence found')
        except StopIteration:
            pass
        self._has_truncated_strings = False
        molecule_type = record.annotations.get('molecule_type')
        if molecule_type is None:
            seqtype = 0
        elif 'DNA' in molecule_type:
            seqtype = 1
        elif 'RNA' in molecule_type:
            seqtype = 3
        elif 'protein' in molecule_type:
            seqtype = 4
        else:
            seqtype = 0
        if record.annotations.get('topology', 'linear') == 'circular':
            topology = 1
        else:
            topology = 0
        if record.description.startswith(record.id):
            comment = record.description
        else:
            comment = f'{record.id} {record.description}'
        self.handle.write(pack('>BBB25xII60xI11xB', 0, seqtype, topology, len(record), 0, len(comment), 255))
        self.handle.write(bytes(record.seq))
        self.handle.write(comment.encode('ASCII'))
        self.handle.write(pack('>B', 0))
        self._write_pstring('0')
        self._write_pstring('0')
        features = [f for f in record.features if isinstance(f.location.start, ExactPosition) and isinstance(f.location.end, ExactPosition)]
        drop = len(record.features) - len(features)
        if drop > 0:
            warnings.warn(f'Dropping {drop} features with fuzzy locations', BiopythonWarning)
        if len(features) > 255:
            drop = len(features) - 255
            warnings.warn(f'Too many features, dropping the last {drop}', BiopythonWarning)
            features = features[:255]
        self.handle.write(pack('>B', len(features)))
        for feature in features:
            self._write_pstring(feature.qualifiers.get('label', [''])[0])
            description = ''
            for qname in feature.qualifiers:
                if qname in ('label', 'translation'):
                    continue
                for val in feature.qualifiers[qname]:
                    if len(description) > 0:
                        description = description + '\r'
                    description = description + f'{qname}="{val}"'
            self._write_pstring(description)
            self._write_pstring(feature.type)
            start = int(feature.location.start) + 1
            end = int(feature.location.end)
            strand = 1
            if feature.location.strand == -1:
                start, end = (end, start)
                strand = 0
            self._write_pstring(str(start))
            self._write_pstring(str(end))
            self.handle.write(pack('>BBBB', strand, 1, 0, 1))
            self._write_pstring('127,127,127')
        if self._has_truncated_strings:
            warnings.warn('Some annotations were truncated to 255 characters', BiopythonWarning)
        return 1

    def _write_pstring(self, s):
        """Write the given string as a Pascal string."""
        if len(s) > 255:
            self._has_truncated_strings = True
            s = s[:255]
        self.handle.write(pack('>B', len(s)))
        self.handle.write(s.encode('ASCII'))