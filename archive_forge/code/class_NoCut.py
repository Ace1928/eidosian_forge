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
class NoCut(AbstractCut):
    """Implement the methods specific to the enzymes that do not cut.

    These enzymes are generally enzymes that have been only partially
    characterised and the way they cut the DNA is unknown or enzymes for
    which the pattern of cut is to complex to be recorded in Rebase
    (ncuts values of 0 in emboss_e.###).

    When using search() with these enzymes the values returned are at the start
    of the restriction site.

    Their catalyse() method returns a TypeError.

    Unknown and NotDefined are also part of the base classes of these enzymes.

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
        return False

    @classmethod
    def _modify(cls, location):
        """Return a generator that moves the cutting position by 1 (PRIVATE).

        For internal use only.

        location is an integer corresponding to the location of the match for
        the enzyme pattern in the sequence.
        _modify returns the real place where the enzyme will cut.

        Example::

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

        If the enzyme cut twice _modify will returns two integer corresponding
        to each cutting site.
        """
        yield location

    @classmethod
    def _rev_modify(cls, location):
        """Return a generator that moves the cutting position by 1 (PRIVATE).

        For internal use only.

        As _modify for site situated on the antiparallel strand when the
        enzyme is not palindromic.
        """
        yield location

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
        return (None, None, None, None, cls.site)