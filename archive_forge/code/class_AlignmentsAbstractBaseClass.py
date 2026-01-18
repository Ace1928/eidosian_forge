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
class AlignmentsAbstractBaseClass(ABC):
    """Abstract base class for sequence alignments.

    Most users will not need to use this class. It is used internally as a base
    class for the list-like Alignments class, and for the AlignmentIterator
    class in Bio.Align.interfaces, which itself is the abstract base class for
    the alignment parsers in Bio/Align/.
    """

    def __iter__(self):
        """Iterate over the alignments as Alignment objects.

        This method SHOULD NOT be overridden by any subclass.
        """
        return self

    @abstractmethod
    def __next__(self):
        """Return the next alignment."""

    @abstractmethod
    def rewind(self):
        """Rewind the iterator to let it loop over the alignments from the beginning."""

    @abstractmethod
    def __len__(self):
        """Return the number of alignments."""