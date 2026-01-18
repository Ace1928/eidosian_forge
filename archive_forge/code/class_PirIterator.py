from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from .Interfaces import _get_seq_string
from .Interfaces import SequenceIterator
from .Interfaces import SequenceWriter
class PirIterator(SequenceIterator):
    """Parser for PIR files."""

    def __init__(self, source):
        """Iterate over a PIR file and yield SeqRecord objects.

        source - file-like object or a path to a file.

        Examples
        --------
        >>> with open("NBRF/DMB_prot.pir") as handle:
        ...    for record in PirIterator(handle):
        ...        print("%s length %i" % (record.id, len(record)))
        HLA:HLA00489 length 263
        HLA:HLA00490 length 94
        HLA:HLA00491 length 94
        HLA:HLA00492 length 80
        HLA:HLA00493 length 175
        HLA:HLA01083 length 188

        """
        super().__init__(source, mode='t', fmt='Pir')

    def parse(self, handle):
        """Start parsing the file, and return a SeqRecord generator."""
        records = self.iterate(handle)
        return records

    def iterate(self, handle):
        """Iterate over the records in the PIR file."""
        for line in handle:
            if line[0] == '>':
                break
        else:
            return
        while True:
            pir_type = line[1:3]
            if pir_type not in _pir_mol_type or line[3] != ';':
                raise ValueError("Records should start with '>XX;' where XX is a valid sequence type")
            identifier = line[4:].strip()
            description = handle.readline().strip()
            lines = []
            for line in handle:
                if line[0] == '>':
                    break
                lines.append(line.rstrip().replace(' ', ''))
            else:
                line = None
            seq = ''.join(lines)
            if seq[-1] != '*':
                raise ValueError('Sequences in PIR files should include a * terminator!')
            record = SeqRecord(Seq(seq[:-1]), id=identifier, name=identifier, description=description)
            record.annotations['PIR-type'] = pir_type
            if _pir_mol_type[pir_type]:
                record.annotations['molecule_type'] = _pir_mol_type[pir_type]
            yield record
            if line is None:
                return
        raise ValueError('Unrecognised PIR record format.')