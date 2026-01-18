from standard phylip format in the following ways:
import string
from Bio.Align import MultipleSeqAlignment
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from .Interfaces import AlignmentIterator
from .Interfaces import SequentialAlignmentWriter

            Quoting the PHYLIP version 3.6 documentation:

            The name should be ten characters in length, filled out to
            the full ten characters by blanks if shorter. Any printable
            ASCII/ISO character is allowed in the name, except for
            parentheses ("(" and ")"), square brackets ("[" and "]"),
            colon (":"), semicolon (";") and comma (","). If you forget
            to extend the names to ten characters in length by blanks,
            the program [i.e. PHYLIP] will get out of synchronization
            with the contents of the data file, and an error message will
            result.

            Note that Tab characters count as only one character in the
            species names. Their inclusion can cause trouble.
            