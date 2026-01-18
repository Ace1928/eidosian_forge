from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from .Interfaces import SequenceIterator
class IgIterator(SequenceIterator):
    """Parser for IntelliGenetics files."""

    def __init__(self, source):
        """Iterate over IntelliGenetics records (as SeqRecord objects).

        source - file-like object opened in text mode, or a path to a file

        The optional free format file header lines (which start with two
        semi-colons) are ignored.

        The free format commentary lines at the start of each record (which
        start with a semi-colon) are recorded as a single string with embedded
        new line characters in the SeqRecord's annotations dictionary under the
        key 'comment'.

        Examples
        --------
        >>> with open("IntelliGenetics/TAT_mase_nuc.txt") as handle:
        ...     for record in IgIterator(handle):
        ...         print("%s length %i" % (record.id, len(record)))
        ...
        A_U455 length 303
        B_HXB2R length 306
        C_UG268A length 267
        D_ELI length 309
        F_BZ163A length 309
        O_ANT70 length 342
        O_MVP5180 length 348
        CPZGAB length 309
        CPZANT length 309
        A_ROD length 390
        B_EHOA length 420
        D_MM251 length 390
        STM_STM length 387
        VER_AGM3 length 354
        GRI_AGM677 length 264
        SAB_SAB1C length 219
        SYK_SYK length 330

        """
        super().__init__(source, mode='t', fmt='IntelliGenetics')

    def parse(self, handle):
        """Start parsing the file, and return a SeqRecord generator."""
        records = self.iterate(handle)
        return records

    def iterate(self, handle):
        """Iterate over the records in the IntelliGenetics file."""
        for line in handle:
            if not line.startswith(';;'):
                break
        else:
            return
        if line[0] != ';':
            raise ValueError(f"Records should start with ';' and not:\n{line!r}")
        while line:
            comment_lines = []
            while line.startswith(';'):
                comment_lines.append(line[1:].strip())
                line = next(handle)
            title = line.rstrip()
            seq_lines = []
            for line in handle:
                if line[0] == ';':
                    break
                seq_lines.append(line.rstrip().replace(' ', ''))
            else:
                line = None
            seq_str = ''.join(seq_lines)
            if seq_str.endswith('1'):
                seq_str = seq_str[:-1]
            if '1' in seq_str:
                raise ValueError('Potential terminator digit one found within sequence.')
            yield SeqRecord(Seq(seq_str), id=title, name=title, annotations={'comment': '\n'.join(comment_lines)})
        assert not line