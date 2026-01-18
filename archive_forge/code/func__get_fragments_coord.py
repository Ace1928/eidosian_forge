import re
from functools import reduce
from abc import ABC, abstractmethod
from typing import Optional, Type
from Bio.SearchIO._index import SearchIndexer
from Bio.SearchIO._model import QueryResult, Hit, HSP, HSPFragment
from Bio.SeqUtils import seq1
def _get_fragments_coord(frags):
    """Return the letter coordinate of the given list of fragments (PRIVATE).

    This function takes a list of three-letter amino acid sequences and
    returns a list of coordinates for each fragment had all the input
    sequences been flattened.

    This is an internal private function and is meant for parsing Exonerate's
    three-letter amino acid output.

    >>> from Bio.SearchIO.ExonerateIO._base import _get_fragments_coord
    >>> _get_fragments_coord(['Thr', 'Ser', 'Ala'])
    [0, 3, 6]
    >>> _get_fragments_coord(['Thr', 'SerAlaPro', 'GlyLeu'])
    [0, 3, 12]
    >>> _get_fragments_coord(['Thr', 'SerAlaPro', 'GlyLeu', 'Cys'])
    [0, 3, 12, 18]

    """
    if not frags:
        return []
    init = [0]
    return reduce(lambda acc, frag: acc + [acc[-1] + len(frag)], frags[:-1], init)