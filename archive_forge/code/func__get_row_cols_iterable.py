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
def _get_row_cols_iterable(self, coordinate, cols, gaps, sequence):
    """Return the alignment contents of one row and multiple columns (PRIVATE).

        This method is called by __getitem__ for invocations of the form

        self[row, cols]

        where row is an integer and cols is an iterable of integers.
        Return value is a string if the aligned sequences are string, Seq,
        or SeqRecord objects, otherwise the return value is a list.
        """
    try:
        sequence = sequence.seq
    except AttributeError:
        pass
    if isinstance(sequence, (str, Seq)):
        line = ''
        start = coordinate[0]
        for end, gap in zip(coordinate[1:], gaps):
            if start < end:
                line += str(sequence[start:end])
            else:
                line += '-' * gap
            start = end
        try:
            line = ''.join((line[col] for col in cols))
        except IndexError:
            raise
        except Exception:
            raise TypeError('second index must be an integer, slice, or iterable of integers') from None
    else:
        line = []
        start = coordinate[0]
        for end, gap in zip(coordinate[1:], gaps):
            if start < end:
                line.extend(sequence[start:end])
            else:
                line.extend([None] * gap)
            start = end
        try:
            line = [line[col] for col in cols]
        except IndexError:
            raise
        except Exception:
            raise TypeError('second index must be an integer, slice, or iterable of integers') from None
    return line