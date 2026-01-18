from Bio.Align import MultipleSeqAlignment
from Bio.Seq import Seq
class AlignmentHasDifferentLengthsError(Exception):
    """Exception where sequences in alignment have different lengths."""