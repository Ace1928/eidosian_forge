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
class PairwiseAligner(_pairwisealigner.PairwiseAligner):
    """Performs pairwise sequence alignment using dynamic programming.

    This provides functions to get global and local alignments between two
    sequences.  A global alignment finds the best concordance between all
    characters in two sequences.  A local alignment finds just the
    subsequences that align the best.

    To perform a pairwise sequence alignment, first create a PairwiseAligner
    object.  This object stores the match and mismatch scores, as well as the
    gap scores.  Typically, match scores are positive, while mismatch scores
    and gap scores are negative or zero.  By default, the match score is 1,
    and the mismatch and gap scores are zero.  Based on the values of the gap
    scores, a PairwiseAligner object automatically chooses the appropriate
    alignment algorithm (the Needleman-Wunsch, Smith-Waterman, Gotoh, or
    Waterman-Smith-Beyer global or local alignment algorithm).

    Calling the "score" method on the aligner with two sequences as arguments
    will calculate the alignment score between the two sequences.
    Calling the "align" method on the aligner with two sequences as arguments
    will return a generator yielding the alignments between the two
    sequences.

    Some examples:

    >>> from Bio import Align
    >>> aligner = Align.PairwiseAligner()
    >>> alignments = aligner.align("TACCG", "ACG")
    >>> for alignment in sorted(alignments):
    ...     print("Score = %.1f:" % alignment.score)
    ...     print(alignment)
    ...
    Score = 3.0:
    target            0 TACCG 5
                      0 -|-|| 5
    query             0 -A-CG 3
    <BLANKLINE>
    Score = 3.0:
    target            0 TACCG 5
                      0 -||-| 5
    query             0 -AC-G 3
    <BLANKLINE>

    Specify the aligner mode as local to generate local alignments:

    >>> aligner.mode = 'local'
    >>> alignments = aligner.align("TACCG", "ACG")
    >>> for alignment in sorted(alignments):
    ...     print("Score = %.1f:" % alignment.score)
    ...     print(alignment)
    ...
    Score = 3.0:
    target            1 ACCG 5
                      0 |-|| 4
    query             0 A-CG 3
    <BLANKLINE>
    Score = 3.0:
    target            1 ACCG 5
                      0 ||-| 4
    query             0 AC-G 3
    <BLANKLINE>

    Do a global alignment.  Identical characters are given 2 points,
    1 point is deducted for each non-identical character.

    >>> aligner.mode = 'global'
    >>> aligner.match_score = 2
    >>> aligner.mismatch_score = -1
    >>> for alignment in aligner.align("TACCG", "ACG"):
    ...     print("Score = %.1f:" % alignment.score)
    ...     print(alignment)
    ...
    Score = 6.0:
    target            0 TACCG 5
                      0 -||-| 5
    query             0 -AC-G 3
    <BLANKLINE>
    Score = 6.0:
    target            0 TACCG 5
                      0 -|-|| 5
    query             0 -A-CG 3
    <BLANKLINE>

    Same as above, except now 0.5 points are deducted when opening a
    gap, and 0.1 points are deducted when extending it.

    >>> aligner.open_gap_score = -0.5
    >>> aligner.extend_gap_score = -0.1
    >>> aligner.target_end_gap_score = 0.0
    >>> aligner.query_end_gap_score = 0.0
    >>> for alignment in aligner.align("TACCG", "ACG"):
    ...     print("Score = %.1f:" % alignment.score)
    ...     print(alignment)
    ...
    Score = 5.5:
    target            0 TACCG 5
                      0 -|-|| 5
    query             0 -A-CG 3
    <BLANKLINE>
    Score = 5.5:
    target            0 TACCG 5
                      0 -||-| 5
    query             0 -AC-G 3
    <BLANKLINE>

    The alignment function can also use known matrices already included in
    Biopython:

    >>> from Bio.Align import substitution_matrices
    >>> aligner = Align.PairwiseAligner()
    >>> aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
    >>> alignments = aligner.align("KEVLA", "EVL")
    >>> alignments = list(alignments)
    >>> print("Number of alignments: %d" % len(alignments))
    Number of alignments: 1
    >>> alignment = alignments[0]
    >>> print("Score = %.1f" % alignment.score)
    Score = 13.0
    >>> print(alignment)
    target            0 KEVLA 5
                      0 -|||- 5
    query             0 -EVL- 3
    <BLANKLINE>

    You can also set the value of attributes directly during construction
    of the PairwiseAligner object by providing them as keyword arguments:

    >>> aligner = Align.PairwiseAligner(mode='global', match_score=2, mismatch_score=-1)
    >>> for alignment in aligner.align("TACCG", "ACG"):
    ...     print("Score = %.1f:" % alignment.score)
    ...     print(alignment)
    ...
    Score = 6.0:
    target            0 TACCG 5
                      0 -||-| 5
    query             0 -AC-G 3
    <BLANKLINE>
    Score = 6.0:
    target            0 TACCG 5
                      0 -|-|| 5
    query             0 -A-CG 3
    <BLANKLINE>

    """

    def __init__(self, scoring=None, **kwargs):
        """Initialize a PairwiseAligner as specified by the keyword arguments.

        If scoring is None, use the default scoring scheme match = 1.0,
        mismatch = 0.0, gap score = 0.0
        If scoring is "blastn", "megablast", or "blastp", use the default
        substitution matrix and gap scores for BLASTN, MEGABLAST, or BLASTP,
        respectively.

        Loops over the remaining keyword arguments and sets them as attributes
        on the object.
        """
        super().__init__()
        if scoring is None:
            pass
        elif scoring == 'blastn':
            self.substitution_matrix = substitution_matrices.load('BLASTN')
            self.open_gap_score = -7.0
            self.extend_gap_score = -2.0
        elif scoring == 'megablast':
            self.substitution_matrix = substitution_matrices.load('MEGABLAST')
            self.open_gap_score = -2.5
            self.extend_gap_score = -2.5
        elif scoring == 'blastp':
            self.substitution_matrix = substitution_matrices.load('BLASTP')
            self.open_gap_score = -12.0
            self.extend_gap_score = -1.0
        else:
            raise ValueError("Unknown scoring scheme '%s'" % scoring)
        for name, value in kwargs.items():
            setattr(self, name, value)

    def __setattr__(self, key, value):
        if key not in dir(_pairwisealigner.PairwiseAligner):
            raise AttributeError("'PairwiseAligner' object has no attribute '%s'" % key)
        _pairwisealigner.PairwiseAligner.__setattr__(self, key, value)

    def align(self, seqA, seqB, strand='+'):
        """Return the alignments of two sequences using PairwiseAligner."""
        if isinstance(seqA, (Seq, MutableSeq, SeqRecord)):
            sA = bytes(seqA)
        else:
            sA = seqA
        if strand == '+':
            sB = seqB
        else:
            sB = reverse_complement(seqB)
        if isinstance(seqB, (Seq, MutableSeq, SeqRecord)):
            sB = bytes(sB)
        score, paths = super().align(sA, sB, strand)
        alignments = PairwiseAlignments(seqA, seqB, score, paths)
        return alignments

    def score(self, seqA, seqB, strand='+'):
        """Return the alignment score of two sequences using PairwiseAligner."""
        if isinstance(seqA, (Seq, MutableSeq, SeqRecord)):
            seqA = bytes(seqA)
        if strand == '-':
            seqB = reverse_complement(seqB)
        if isinstance(seqB, (Seq, MutableSeq, SeqRecord)):
            seqB = bytes(seqB)
        return super().score(seqA, seqB, strand)

    def __getstate__(self):
        state = {'wildcard': self.wildcard, 'target_internal_open_gap_score': self.target_internal_open_gap_score, 'target_internal_extend_gap_score': self.target_internal_extend_gap_score, 'target_left_open_gap_score': self.target_left_open_gap_score, 'target_left_extend_gap_score': self.target_left_extend_gap_score, 'target_right_open_gap_score': self.target_right_open_gap_score, 'target_right_extend_gap_score': self.target_right_extend_gap_score, 'query_internal_open_gap_score': self.query_internal_open_gap_score, 'query_internal_extend_gap_score': self.query_internal_extend_gap_score, 'query_left_open_gap_score': self.query_left_open_gap_score, 'query_left_extend_gap_score': self.query_left_extend_gap_score, 'query_right_open_gap_score': self.query_right_open_gap_score, 'query_right_extend_gap_score': self.query_right_extend_gap_score, 'mode': self.mode}
        if self.substitution_matrix is None:
            state['match_score'] = self.match_score
            state['mismatch_score'] = self.mismatch_score
        else:
            state['substitution_matrix'] = self.substitution_matrix
        return state

    def __setstate__(self, state):
        self.wildcard = state['wildcard']
        self.target_internal_open_gap_score = state['target_internal_open_gap_score']
        self.target_internal_extend_gap_score = state['target_internal_extend_gap_score']
        self.target_left_open_gap_score = state['target_left_open_gap_score']
        self.target_left_extend_gap_score = state['target_left_extend_gap_score']
        self.target_right_open_gap_score = state['target_right_open_gap_score']
        self.target_right_extend_gap_score = state['target_right_extend_gap_score']
        self.query_internal_open_gap_score = state['query_internal_open_gap_score']
        self.query_internal_extend_gap_score = state['query_internal_extend_gap_score']
        self.query_left_open_gap_score = state['query_left_open_gap_score']
        self.query_left_extend_gap_score = state['query_left_extend_gap_score']
        self.query_right_open_gap_score = state['query_right_open_gap_score']
        self.query_right_extend_gap_score = state['query_right_extend_gap_score']
        self.mode = state['mode']
        substitution_matrix = state.get('substitution_matrix')
        if substitution_matrix is None:
            self.match_score = state['match_score']
            self.mismatch_score = state['mismatch_score']
        else:
            self.substitution_matrix = substitution_matrix