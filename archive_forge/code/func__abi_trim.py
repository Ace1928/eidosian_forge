import datetime
import struct
import sys
from os.path import basename
from typing import List
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from .Interfaces import SequenceIterator
def _abi_trim(seq_record):
    """Trims the sequence using Richard Mott's modified trimming algorithm (PRIVATE).

    Arguments:
        - seq_record - SeqRecord object to be trimmed.

    Trimmed bases are determined from their segment score, which is a
    cumulative sum of each base's score. Base scores are calculated from
    their quality values.

    More about the trimming algorithm:
    http://www.phrap.org/phredphrap/phred.html
    http://resources.qiagenbioinformatics.com/manuals/clcgenomicsworkbench/650/Quality_trimming.html
    """
    start = False
    segment = 20
    trim_start = 0
    cutoff = 0.05
    if len(seq_record) <= segment:
        return seq_record
    else:
        score_list = [cutoff - 10 ** (qual / -10.0) for qual in seq_record.letter_annotations['phred_quality']]
        cummul_score = [0]
        for i in range(1, len(score_list)):
            score = cummul_score[-1] + score_list[i]
            if score < 0:
                cummul_score.append(0)
            else:
                cummul_score.append(score)
                if not start:
                    trim_start = i
                    start = True
        trim_finish = cummul_score.index(max(cummul_score))
        return seq_record[trim_start:trim_finish]