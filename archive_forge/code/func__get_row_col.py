import sys
import collections
import copy
import importlib
import types
import warnings
import numbers
from itertools import zip_longest
from abc import ABC, abstractmethod
from typing import Dict
from Bio.Align import _pairwisealigner  # type: ignore
from Bio.Align import _codonaligner  # type: ignore
from Bio.Align import substitution_matrices
from Bio.Data import CodonTable
from Bio.Seq import Seq, MutableSeq, reverse_complement, UndefinedSequenceError
from Bio.Seq import translate
from Bio.SeqRecord import SeqRecord, _RestrictedDict
def _get_row_col(self, j, col, steps, gaps, sequence):
    """Return the sequence contents at alignment column j (PRIVATE).

        This method is called by __getitem__ for invocations of the form

        self[row, col]

        where both row and col are integers.
        Return value is a string of length 1.
        """
    indices = gaps.cumsum()
    index = indices.searchsorted(col, side='right')
    if steps[index]:
        offset = col - indices[index]
        j += sum(steps[:index + 1]) + offset
        return sequence[j]
    else:
        return '-'