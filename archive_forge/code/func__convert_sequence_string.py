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
def _convert_sequence_string(self, sequence):
    """Convert given sequence to string using the appropriate method (PRIVATE)."""
    if isinstance(sequence, (bytes, bytearray)):
        return sequence.decode()
    if isinstance(sequence, str):
        return sequence
    if isinstance(sequence, Seq):
        return str(sequence)
    try:
        sequence = sequence.seq
    except AttributeError:
        pass
    else:
        return str(sequence)
    try:
        view = memoryview(sequence)
    except TypeError:
        pass
    else:
        if view.format == 'c':
            return str(sequence)
    return None