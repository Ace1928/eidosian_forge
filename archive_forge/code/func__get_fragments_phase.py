import re
from functools import reduce
from abc import ABC, abstractmethod
from typing import Optional, Type
from Bio.SearchIO._index import SearchIndexer
from Bio.SearchIO._model import QueryResult, Hit, HSP, HSPFragment
from Bio.SeqUtils import seq1
def _get_fragments_phase(frags):
    """Return the phases of the given list of 3-letter amino acid fragments (PRIVATE).

    This is an internal private function and is meant for parsing Exonerate's
    three-letter amino acid output.

    >>> from Bio.SearchIO.ExonerateIO._base import _get_fragments_phase
    >>> _get_fragments_phase(['Thr', 'Ser', 'Ala'])
    [0, 0, 0]
    >>> _get_fragments_phase(['ThrSe', 'rAla'])
    [0, 1]
    >>> _get_fragments_phase(['ThrSe', 'rAlaLeu', 'ProCys'])
    [0, 1, 0]
    >>> _get_fragments_phase(['ThrSe', 'rAlaLeuP', 'roCys'])
    [0, 1, 2]
    >>> _get_fragments_phase(['ThrSe', 'rAlaLeuPr', 'oCys'])
    [0, 1, 1]

    """
    return [(3 - x % 3) % 3 for x in _get_fragments_coord(frags)]