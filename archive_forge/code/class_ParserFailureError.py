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
class ParserFailureError(ValueError):
    """Failure caused by some kind of problem in the parser."""