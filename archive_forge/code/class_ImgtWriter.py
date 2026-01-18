import warnings
from datetime import datetime
from Bio import BiopythonWarning
from Bio import SeqFeature
from Bio import SeqIO
from Bio.GenBank.Scanner import _ImgtScanner
from Bio.GenBank.Scanner import EmblScanner
from Bio.GenBank.Scanner import GenBankScanner
from Bio.Seq import UndefinedSequenceError
from .Interfaces import _get_seq_string
from .Interfaces import SequenceIterator
from .Interfaces import SequenceWriter
class ImgtWriter(EmblWriter):
    """IMGT writer (EMBL format variant)."""
    HEADER_WIDTH = 5
    QUALIFIER_INDENT = 25
    QUALIFIER_INDENT_STR = 'FT' + ' ' * (QUALIFIER_INDENT - 2)
    QUALIFIER_INDENT_TMP = 'FT   %s                    '
    FEATURE_HEADER = 'FH   Key                 Location/Qualifiers\nFH\n'