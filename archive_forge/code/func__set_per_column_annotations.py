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
def _set_per_column_annotations(self, value):
    if not isinstance(value, dict):
        raise TypeError('The per-column-annotations should be a (restricted) dictionary.')
    if len(self):
        expected_length = self.get_alignment_length()
        self._per_col_annotations = _RestrictedDict(length=expected_length)
        self._per_col_annotations.update(value)
    else:
        self._per_col_annotations = None
        if value:
            raise ValueError("Can't set per-column-annotations without an alignment")