import warnings
import re
import string
import itertools
from Bio.Seq import Seq, MutableSeq
from Bio.Restriction.Restriction_Dictionary import rest_dict as enzymedict
from Bio.Restriction.Restriction_Dictionary import typedict
from Bio.Restriction.Restriction_Dictionary import suppliers as suppliers_dict
from Bio.Restriction.PrintFormat import PrintFormat
from Bio import BiopythonWarning
class TwoCuts(AbstractCut):
    """Implement the methods for enzymes that cut the DNA twice.

    Correspond to ncuts values of 4 in emboss_e.###

    Internal use only. Not meant to be instantiated.
    """

    @classmethod
    def cut_once(cls):
        """Return if the cutting pattern has one cut.

        True if the enzyme cut the sequence one time on each strand.
        """
        return False

    @classmethod
    def cut_twice(cls):
        """Return if the cutting pattern has two cuts.

        True if the enzyme cut the sequence twice on each strand.
        """
        return True

    @classmethod
    def _modify(cls, location):
        """Return a generator that moves the cutting position by 1 (PRIVATE).

        For internal use only.

        location is an integer corresponding to the location of the match for
        the enzyme pattern in the sequence.
        _modify returns the real place where the enzyme will cut.

        example::

            EcoRI pattern : GAATTC
            EcoRI will cut after the G.
            so in the sequence:
                     ______
            GAATACACGGAATTCGA
                     |
                     10
            dna.finditer(GAATTC, 6) will return 10 as G is the 10th base
            EcoRI cut after the G so:
            EcoRI._modify(10) -> 11.

        if the enzyme cut twice _modify will returns two integer corresponding
        to each cutting site.
        """
        yield (location + cls.fst5)
        yield (location + cls.scd5)

    @classmethod
    def _rev_modify(cls, location):
        """Return a generator that moves the cutting position by 1 (PRIVATE).

        for internal use only.

        as _modify for site situated on the antiparallel strand when the
        enzyme is not palindromic
        """
        yield (location - cls.fst3)
        yield (location - cls.scd3)

    @classmethod
    def characteristic(cls):
        """Return a list of the enzyme's characteristics as tuple.

        the tuple contains the attributes:

        - fst5 -> first 5' cut ((current strand) or None
        - fst3 -> first 3' cut (complementary strand) or None
        - scd5 -> second 5' cut (current strand) or None
        - scd5 -> second 3' cut (complementary strand) or None
        - site -> recognition site.

        """
        return (cls.fst5, cls.fst3, cls.scd5, cls.scd3, cls.site)