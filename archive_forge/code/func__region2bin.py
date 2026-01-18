import os
from itertools import islice
from Bio.Align import MultipleSeqAlignment
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from .Interfaces import SequentialAlignmentWriter
@staticmethod
def _region2bin(start, end):
    """Find bins that a region may belong to (PRIVATE).

        Converts a region to a list of bins that it may belong to, including largest
        and smallest bins.
        """
    bins = [0, 1]
    bins.extend(range(1 + (start >> 26), 2 + (end - 1 >> 26)))
    bins.extend(range(9 + (start >> 23), 10 + (end - 1 >> 23)))
    bins.extend(range(73 + (start >> 20), 74 + (end - 1 >> 20)))
    bins.extend(range(585 + (start >> 17), 586 + (end - 1 >> 17)))
    return set(bins)