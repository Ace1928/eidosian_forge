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
def _split_accessions(accession_string):
    """Split a string of accession numbers into a list (PRIVATE)."""
    accession = accession_string.replace('\n', ' ').replace(';', ' ')
    return [x.strip() for x in accession.split() if x.strip()]