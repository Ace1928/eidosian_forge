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
def _format_generalized(self):
    """Return generalized string representation (PRIVATE).

        Helper for self._format_pretty().
        """
    seq1, seq2 = self.sequences
    aligned_seq1 = []
    aligned_seq2 = []
    pattern = []
    end1, end2 = self.coordinates[:, 0]
    if end1 > 0 or end2 > 0:
        if end1 <= end2:
            for c2 in seq2[:end2 - end1]:
                s2 = str(c2)
                s1 = ' ' * len(s2)
                aligned_seq1.append(s1)
                aligned_seq2.append(s2)
                pattern.append(s1)
        else:
            for c1 in seq1[:end1 - end2]:
                s1 = str(c1)
                s2 = ' ' * len(s1)
                aligned_seq1.append(s1)
                aligned_seq2.append(s2)
                pattern.append(s2)
    start1 = end1
    start2 = end2
    for end1, end2 in self.coordinates[:, 1:].transpose():
        if end1 == start1:
            for c2 in seq2[start2:end2]:
                s2 = str(c2)
                s1 = '-' * len(s2)
                aligned_seq1.append(s1)
                aligned_seq2.append(s2)
                pattern.append(s1)
            start2 = end2
        elif end2 == start2:
            for c1 in seq1[start1:end1]:
                s1 = str(c1)
                s2 = '-' * len(s1)
                aligned_seq1.append(s1)
                aligned_seq2.append(s2)
                pattern.append(s2)
            start1 = end1
        else:
            t1 = seq1[start1:end1]
            t2 = seq2[start2:end2]
            if len(t1) != len(t2):
                raise ValueError('Unequal step sizes in alignment')
            for c1, c2 in zip(t1, t2):
                s1 = str(c1)
                s2 = str(c2)
                m1 = len(s1)
                m2 = len(s2)
                if c1 == c2:
                    p = '|'
                else:
                    p = '.'
                if m1 < m2:
                    space = (m2 - m1) * ' '
                    s1 += space
                    pattern.append(p * m1 + space)
                elif m1 > m2:
                    space = (m1 - m2) * ' '
                    s2 += space
                    pattern.append(p * m2 + space)
                else:
                    pattern.append(p * m1)
                aligned_seq1.append(s1)
                aligned_seq2.append(s2)
            start1 = end1
            start2 = end2
    aligned_seq1 = ' '.join(aligned_seq1)
    aligned_seq2 = ' '.join(aligned_seq2)
    pattern = ' '.join(pattern)
    return f'{aligned_seq1}\n{pattern}\n{aligned_seq2}\n'