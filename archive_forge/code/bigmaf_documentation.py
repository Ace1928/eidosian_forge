from io import StringIO
from Bio.Align import Alignment, Alignments
from Bio.Align import interfaces, bigbed, maf
from Bio.Align.bigbed import AutoSQLTable, Field
from Bio.SeqRecord import SeqRecord
Create an AlignmentIterator object.

        Arguments:
        - source - input file stream, or path to input file
        