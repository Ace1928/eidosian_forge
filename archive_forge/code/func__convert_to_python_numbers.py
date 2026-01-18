import re
import warnings
from Bio import BiopythonParserWarning
from Bio.Seq import Seq
from Bio.SeqFeature import Location
from Bio.SeqFeature import Reference
from Bio.SeqFeature import SeqFeature
from Bio.SeqFeature import SimpleLocation
from Bio.SeqFeature import LocationParserError
from .utils import FeatureValueCleaner
from .Scanner import GenBankScanner
@staticmethod
def _convert_to_python_numbers(start, end):
    """Convert a start and end range to python notation (PRIVATE).

        In GenBank, starts and ends are defined in "biological" coordinates,
        where 1 is the first base and [i, j] means to include both i and j.

        In python, 0 is the first base and [i, j] means to include i, but
        not j.

        So, to convert "biological" to python coordinates, we need to
        subtract 1 from the start, and leave the end and things should
        be converted happily.
        """
    new_start = start - 1
    new_end = end
    return (new_start, new_end)