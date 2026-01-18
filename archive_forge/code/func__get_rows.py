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
def _get_rows(self, key):
    """Return self[key], where key is a slice object (PRIVATE).

        This method is called by __getitem__ for invocations of the form

        self[rows]

        where rows is a slice object. Return value is an Alignment object.
        """
    sequences = self.sequences[key]
    coordinates = self.coordinates[key].copy()
    alignment = Alignment(sequences, coordinates)
    if np.array_equal(self.coordinates, coordinates):
        try:
            alignment.score = self.score
        except AttributeError:
            pass
        try:
            alignment.column_annotations = self.column_annotations
        except AttributeError:
            pass
    return alignment