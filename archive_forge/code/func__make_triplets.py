import re
from functools import reduce
from abc import ABC, abstractmethod
from typing import Optional, Type
from Bio.SearchIO._index import SearchIndexer
from Bio.SearchIO._model import QueryResult, Hit, HSP, HSPFragment
from Bio.SeqUtils import seq1
def _make_triplets(seq, phase=0):
    """Select a valid amino acid sequence given a 3-letter code input (PRIVATE).

    This function takes a single three-letter amino acid sequence and the phase
    of the sequence to return the longest intact amino acid sequence possible.
    Parts of the input sequence before and after the selected sequence are also
    returned.

    This is an internal private function and is meant for parsing Exonerate's
    three-letter amino acid output.

    >>> from Bio.SearchIO.ExonerateIO._base import _make_triplets
    >>> _make_triplets('GlyThrSerAlaPro')
    ('', ['Gly', 'Thr', 'Ser', 'Ala', 'Pro'], '')
    >>> _make_triplets('yThrSerAla', phase=1)
    ('y', ['Thr', 'Ser', 'Ala'], '')
    >>> _make_triplets('yThrSerAlaPr', phase=1)
    ('y', ['Thr', 'Ser', 'Ala'], 'Pr')

    """
    pre = seq[:phase]
    np_seq = seq[phase:]
    non_triplets = len(np_seq) % 3
    post = '' if not non_triplets else np_seq[-1 * non_triplets:]
    intacts = [np_seq[3 * i:3 * (i + 1)] for i in range(len(np_seq) // 3)]
    return (pre, intacts, post)