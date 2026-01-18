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
def _get_per_column_annotations(self):
    if self._per_col_annotations is None:
        if len(self):
            expected_length = self.get_alignment_length()
        else:
            expected_length = 0
        self._per_col_annotations = _RestrictedDict(length=expected_length)
    return self._per_col_annotations