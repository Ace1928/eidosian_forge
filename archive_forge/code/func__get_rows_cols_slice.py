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
def _get_rows_cols_slice(self, coordinates, row, start_index, stop_index, steps, gaps):
    """Return a subalignment of multiple rows and consecutive columns (PRIVATE).

        This method is called by __getitem__ for invocations of the form

        self[rows, cols]

        where rows is an arbitrary slice object, and cols is a slice object
        with step 1, allowing the alignment sequences to be reused in the
        subalignment. Return value is an Alignment object.
        """
    rcs = np.any(coordinates != self.coordinates[row], axis=1)
    indices = gaps.cumsum()
    i = indices.searchsorted(start_index, side='right')
    j = i + indices[i:].searchsorted(stop_index, side='left') + 1
    offset = steps[:, i] - indices[i] + start_index
    coordinates[:, i] += offset * (steps[:, i] > 0)
    offset = indices[j - 1] - stop_index
    coordinates[:, j] -= offset * (steps[:, j - 1] > 0)
    coordinates = coordinates[:, i:j + 1]
    sequences = self.sequences[row]
    for coordinate, rc, sequence in zip(coordinates, rcs, sequences):
        if rc:
            coordinate[:] = len(sequence) - coordinate[:]
    alignment = Alignment(sequences, coordinates)
    if np.array_equal(self.coordinates, coordinates):
        try:
            alignment.score = self.score
        except AttributeError:
            pass
    try:
        column_annotations = self.column_annotations
    except AttributeError:
        pass
    else:
        alignment.column_annotations = {}
        for key, value in column_annotations.items():
            value = value[start_index:stop_index]
            try:
                value = value.copy()
            except AttributeError:
                pass
            alignment.column_annotations[key] = value
    return alignment