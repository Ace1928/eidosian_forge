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
def _get_row_cols_slice(self, coordinate, start_index, stop_index, steps, gaps, sequence):
    """Return the alignment contents of one row and consecutive columns (PRIVATE).

        This method is called by __getitem__ for invocations of the form

        self[row, cols]

        where row is an integer and cols is a slice object with step 1.
        Return value is a string if the aligned sequences are string, Seq,
        or SeqRecord objects, otherwise the return value is a list.
        """
    indices = gaps.cumsum()
    i = indices.searchsorted(start_index, side='right')
    j = i + indices[i:].searchsorted(stop_index, side='right')
    try:
        sequence = sequence.seq
    except AttributeError:
        pass
    if isinstance(sequence, (str, Seq)):
        if i == j:
            length = stop_index - start_index
            if steps[i] == 0:
                line = '-' * length
            else:
                start = coordinate[i] + start_index - indices[i - 1]
                stop = start + length
                line = str(sequence[start:stop])
        else:
            length = indices[i] - start_index
            if steps[i] == 0:
                line = '-' * length
            else:
                stop = coordinate[i + 1]
                start = stop - length
                line = str(sequence[start:stop])
            i += 1
            while i < j:
                step = gaps[i]
                if steps[i] == 0:
                    line += '-' * step
                else:
                    start = coordinate[i]
                    stop = coordinate[i + 1]
                    line += str(sequence[start:stop])
                i += 1
            length = stop_index - indices[i - 1]
            if length > 0:
                if steps[i] == 0:
                    line += '-' * length
                else:
                    start = coordinate[i]
                    stop = start + length
                    line += str(sequence[start:stop])
    elif i == j:
        length = stop_index - start_index
        if steps[i] == 0:
            line = [None] * length
        else:
            start = coordinate[i] + start_index - indices[i - 1]
            stop = start + length
            line = sequence[start:stop]
    else:
        length = indices[i] - start_index
        if steps[i] == 0:
            line = [None] * length
        else:
            stop = coordinate[i + 1]
            start = stop - length
            line = sequence[start:stop]
        i += 1
        while i < j:
            step = gaps[i]
            if steps[i] == 0:
                line.extend([None] * step)
            else:
                start = coordinate[i]
                stop = coordinate[i + 1]
                line.extend(sequence[start:stop])
            i += 1
        length = stop_index - indices[i - 1]
        if length > 0:
            if steps[j] == 0:
                line.extend([None] * length)
            else:
                start = coordinate[i]
                stop = start + length
                line.extend(sequence[start:stop])
    return line