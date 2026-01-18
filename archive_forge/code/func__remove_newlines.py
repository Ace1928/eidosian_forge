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
def _remove_newlines(text):
    """Remove any newlines in the passed text, returning the new string (PRIVATE)."""
    newlines = ['\n', '\r']
    for ws in newlines:
        text = text.replace(ws, '')
    return text