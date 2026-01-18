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
class CodonAligner(_codonaligner.CodonAligner):
    """Aligns a nucleotide sequence to an amino acid sequence.

    This class implements a dynamic programming algorithm to align a nucleotide
    sequence to an amino acid sequence.
    """

    def __init__(self, codon_table=None, anchor_len=10):
        """Initialize a CodonAligner for a specific genetic code.

        Arguments:
         - codon_table - a CodonTable object representing the genetic code.
           If codon_table is None, the standard genetic code is used.

        """
        super().__init__()
        if codon_table is None:
            codon_table = CodonTable.generic_by_id[1]
        elif not isinstance(codon_table, CodonTable.CodonTable):
            raise TypeError('Input table is not a CodonTable object')
        self.codon_table = codon_table

    def score(self, seqA, seqB):
        """Return the alignment score of a protein sequence and nucleotide sequence.

        Arguments:
         - seqA  - the protein sequence of amino acids (plain string, Seq,
           MutableSeq, or SeqRecord).
         - seqB  - the nucleotide sequence (plain string, Seq, MutableSeq, or
           SeqRecord); both DNA and RNA sequences are accepted.

        >>> from Bio.Seq import Seq
        >>> from Bio.SeqRecord import SeqRecord
        >>> aligner = CodonAligner()
        >>> dna = SeqRecord(Seq('ATGTCTCGT'), id='dna')
        >>> pro = SeqRecord(Seq('MSR'), id='pro')
        >>> score = aligner.score(pro, dna)
        >>> print(score)
        3.0
        >>> rna = SeqRecord(Seq('AUGUCUCGU'), id='rna')
        >>> score = aligner.score(pro, rna)
        >>> print(score)
        3.0

        This is an example with a frame shift in the DNA sequence:

        >>> dna = "ATGCTGGGCTCGAACGAGTCCGTGTATGCCCTAAGCTGAGCCCGTCG"
        >>> pro = "MLGSNESRVCPKLSPS"
        >>> len(pro)
        16
        >>> aligner.frameshift_score = -3.0
        >>> score = aligner.score(pro, dna)
        >>> print(score)
        13.0

        In the following example, the position of the frame shift is ambiguous:

        >>> dna = 'TTTAAAAAAAAAAATTT'
        >>> pro = 'FKKKKF'
        >>> len(pro)
        6
        >>> aligner.frameshift_score = -1.0
        >>> alignments = aligner.align(pro, dna)
        >>> print(alignments.score)
        5.0
        >>> len(alignments)
        3
        >>> print(next(alignments))
        target            0 F  K  K  K   4
        query             0 TTTAAAAAAAAA 12
        <BLANKLINE>
        target            4 K  F    6
        query            11 AAATTT 17
        <BLANKLINE>
        >>> print(next(alignments))
        target            0 F  K  K   3
        query             0 TTTAAAAAA 9
        <BLANKLINE>
        target            3 K  K  F    6
        query             8 AAAAAATTT 17
        <BLANKLINE>
        >>> print(next(alignments))
        target            0 F  K   2
        query             0 TTTAAA 6
        <BLANKLINE>
        target            2 K  K  K  F    6
        query             5 AAAAAAAAATTT 17
        <BLANKLINE>
        >>> print(next(alignments))
        Traceback (most recent call last):
        ...
        StopIteration

        """
        codon_table = self.codon_table
        if isinstance(seqA, (Seq, MutableSeq, SeqRecord)):
            sA = bytes(seqA)
        elif isinstance(seqA, str):
            sA = seqA.encode()
        else:
            raise ValueError('seqA must be a string, Seq, MutableSeq, or SeqRecord object')
        seqB0 = seqB[:3 * (len(seqB) // 3)]
        seqB1 = seqB[1:1 + 3 * ((len(seqB) - 1) // 3)]
        seqB2 = seqB[2:2 + 3 * ((len(seqB) - 2) // 3)]
        if isinstance(seqB, (Seq, MutableSeq, SeqRecord)):
            sB0 = seqB0.translate(codon_table)
            sB1 = seqB1.translate(codon_table)
            sB2 = seqB2.translate(codon_table)
            sB0 = bytes(sB0)
            sB1 = bytes(sB1)
            sB2 = bytes(sB2)
        elif isinstance(seqA, str):
            sB0 = translate(seqB0, codon_table)
            sB1 = translate(seqB1, codon_table)
            sB2 = translate(seqB2, codon_table)
            sB0 = sB0.encode()
            sB1 = sB1.encode()
            sB2 = sB2.encode()
        else:
            raise ValueError('seqB must be a string, Seq, MutableSeq, or SeqRecord object')
        return super().score(sA, sB0, sB1, sB2)

    def align(self, seqA, seqB):
        """Align a nucleotide sequence to its corresponding protein sequence.

        Arguments:
         - seqA  - the protein sequence of amino acids (plain string, Seq,
           MutableSeq, or SeqRecord).
         - seqB  - the nucleotide sequence (plain string, Seq, MutableSeq, or
           SeqRecord); both DNA and RNA sequences are accepted.

        Returns an iterator of Alignment objects.

        >>> from Bio.Seq import Seq
        >>> from Bio.SeqRecord import SeqRecord
        >>> aligner = CodonAligner()
        >>> dna = SeqRecord(Seq('ATGTCTCGT'), id='dna')
        >>> pro = SeqRecord(Seq('MSR'), id='pro')
        >>> alignments = aligner.align(pro, dna)
        >>> alignment = alignments[0]
        >>> print(alignment)
        pro               0 M  S  R   3
        dna               0 ATGTCTCGT 9
        <BLANKLINE>
        >>> rna = SeqRecord(Seq('AUGUCUCGU'), id='rna')
        >>> alignments = aligner.align(pro, rna)
        >>> alignment = alignments[0]
        >>> print(alignment)
        pro               0 M  S  R   3
        rna               0 AUGUCUCGU 9
        <BLANKLINE>

        This is an example with a frame shift in the DNA sequence:

        >>> dna = "ATGCTGGGCTCGAACGAGTCCGTGTATGCCCTAAGCTGAGCCCGTCG"
        >>> pro = "MLGSNESRVCPKLSPS"
        >>> alignments = aligner.align(pro, dna)
        >>> alignment = alignments[0]
        >>> print(alignment)
        target            0 M  L  G  S  N  E  S   7
        query             0 ATGCTGGGCTCGAACGAGTCC 21
        <BLANKLINE>
        target            7 R  V  C  P  K  L  S  P  S   16
        query            20 CGTGTATGCCCTAAGCTGAGCCCGTCG 47
        <BLANKLINE>

        """
        codon_table = self.codon_table
        if isinstance(seqA, (Seq, MutableSeq, SeqRecord)):
            sA = bytes(seqA)
        elif isinstance(seqA, str):
            sA = seqA.encode()
        else:
            raise ValueError('seqA must be a string, Seq, MutableSeq, or SeqRecord object')
        seqB0 = seqB[:3 * (len(seqB) // 3)]
        seqB1 = seqB[1:1 + 3 * ((len(seqB) - 1) // 3)]
        seqB2 = seqB[2:2 + 3 * ((len(seqB) - 2) // 3)]
        if isinstance(seqB, (Seq, MutableSeq, SeqRecord)):
            sB0 = seqB0.translate(codon_table)
            sB1 = seqB1.translate(codon_table)
            sB2 = seqB2.translate(codon_table)
            sB0 = bytes(sB0)
            sB1 = bytes(sB1)
            sB2 = bytes(sB2)
        elif isinstance(seqA, str):
            sB0 = translate(seqB0, codon_table)
            sB1 = translate(seqB1, codon_table)
            sB2 = translate(seqB2, codon_table)
            sB0 = sB0.encode()
            sB1 = sB1.encode()
            sB2 = sB2.encode()
        else:
            raise ValueError('seqB must be a string, Seq, MutableSeq, or SeqRecord object')
        score, paths = super().align(sA, sB0, sB1, sB2)
        alignments = PairwiseAlignments(seqA, seqB, score, paths)
        return alignments