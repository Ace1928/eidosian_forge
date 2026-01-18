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
def _clean_location(location_string):
    """Clean whitespace out of a location string (PRIVATE).

        The location parser isn't a fan of whitespace, so we clean it out
        before feeding it into the parser.
        """
    return ''.join(location_string.split())