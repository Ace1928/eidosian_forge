import xml.etree.ElementTree as ET
from Bio import Align
from Bio import Seq
from Bio import motifs
def __convert_strand(strand):
    """Convert strand (+/-) from XML if present.

    Default: +
    """
    if strand == 'minus':
        return '-'
    if strand == 'plus' or strand == 'none':
        return '+'