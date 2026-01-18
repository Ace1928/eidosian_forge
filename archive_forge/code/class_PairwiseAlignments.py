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
class PairwiseAlignments(AlignmentsAbstractBaseClass):
    """Implements an iterator over pairwise alignments returned by the aligner.

    This class also supports indexing, which is fast for increasing indices,
    but may be slow for random access of a large number of alignments.

    Note that pairwise aligners can return an astronomical number of alignments,
    even for relatively short sequences, if they align poorly to each other. We
    therefore recommend to first check the number of alignments, accessible as
    len(alignments), which can be calculated quickly even if the number of
    alignments is very large.
    """

    def __init__(self, seqA, seqB, score, paths):
        """Initialize a new PairwiseAlignments object.

        Arguments:
         - seqA  - The first sequence, as a plain string, without gaps.
         - seqB  - The second sequence, as a plain string, without gaps.
         - score - The alignment score.
         - paths - An iterator over the paths in the traceback matrix;
                   each path defines one alignment.

        You would normally obtain a PairwiseAlignments object by calling
        aligner.align(seqA, seqB), where aligner is a PairwiseAligner object
        or a CodonAligner object.
        """
        self.sequences = [seqA, seqB]
        self.score = score
        self._paths = paths
        self._index = -1

    def __len__(self):
        return len(self._paths)

    def __iter__(self):
        self.rewind()
        return self

    def __getitem__(self, index):
        if not isinstance(index, int):
            raise TypeError(f'index must be an integer, not {index.__class__.__name__}')
        if index < 0:
            index += len(self._paths)
        if index == self._index:
            return self._alignment
        if index < self._index:
            self._paths.reset()
            self._index = -1
        while True:
            try:
                alignment = next(self)
            except StopIteration:
                raise IndexError('index out of range') from None
            if self._index == index:
                break
        return alignment

    def __next__(self):
        path = next(self._paths)
        self._index += 1
        coordinates = np.array(path)
        alignment = Alignment(self.sequences, coordinates)
        alignment.score = self.score
        self._alignment = alignment
        return alignment

    def rewind(self):
        self._paths.reset()
        self._index = -1