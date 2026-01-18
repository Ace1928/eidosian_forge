import re
from functools import reduce
from abc import ABC, abstractmethod
from typing import Optional, Type
from Bio.SearchIO._index import SearchIndexer
from Bio.SearchIO._model import QueryResult, Hit, HSP, HSPFragment
from Bio.SeqUtils import seq1
def _get_strand_from_desc(desc, is_protein, modify_desc=True):
    """Determine the strand from the description (PRIVATE).

    Exonerate appends ``:[revcomp]`` (versions <= 2.2) or ``[revcomp]``
    (versions > 2.2) to the query and/or hit description string. This function
    outputs '-' if the description has such modifications or '+' if not. If the
    query and/or hit is a protein sequence, a '.' is output instead.

    Aside from the strand, the input description value is also returned. It is
    returned unmodified if ``modify_desc`` is ``False``. Otherwise, the appended
    ``:[revcomp]`` or ``[revcomp]`` is removed.

    """
    if is_protein:
        return ('.', desc)
    suffix = ''
    if desc.endswith('[revcomp]'):
        suffix = ':[revcomp]' if desc.endswith(':[revcomp]') else '[revcomp]'
    if not suffix:
        return ('+', desc)
    if modify_desc:
        return ('-', desc[:-len(suffix)])
    return ('-', desc)