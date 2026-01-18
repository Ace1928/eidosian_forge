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
def _get_rows_cols_iterable(self, coordinates, col, steps, gaps, sequences):
    """Return a subalignment of multiple rows and columns (PRIVATE).

        This method is called by __getitem__ for invocations of the form

        self[rows, cols]

        where rows is a slice object and cols is an iterable of integers.
        This method will create new sequences for use by the subalignment
        object. Return value is an Alignment object.
        """
    indices = tuple(col)
    lines = []
    for i, sequence in enumerate(sequences):
        try:
            s = sequence.seq
        except AttributeError:
            s = sequence
        line = ''
        k = coordinates[i, 0]
        for step, gap in zip(steps[i], gaps):
            if step:
                j = k + step
                line += str(s[k:j])
                k = j
            else:
                line += '-' * gap
        try:
            line = ''.join((line[index] for index in indices))
        except IndexError:
            raise
        except Exception:
            raise TypeError('second index must be an integer, slice, or iterable of integers') from None
        lines.append(line)
        line = line.replace('-', '')
        s = s.__class__(line)
        try:
            sequence.seq
        except AttributeError:
            sequence = s
        else:
            sequence = copy.deepcopy(sequence)
            sequence.seq = s
        sequences[i] = sequence
    coordinates = self.infer_coordinates(lines)
    alignment = Alignment(sequences, coordinates)
    try:
        column_annotations = self.column_annotations
    except AttributeError:
        pass
    else:
        alignment.column_annotations = {}
        for key, value in column_annotations.items():
            values = (value[index] for index in indices)
            if isinstance(value, str):
                value = ''.join(values)
            else:
                value = value.__class__(values)
            alignment.column_annotations[key] = value
    return alignment