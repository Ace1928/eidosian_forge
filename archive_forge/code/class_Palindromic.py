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
class Palindromic(AbstractCut):
    """Implement methods for enzymes with palindromic recognition sites.

    palindromic means : the recognition site and its reverse complement are
                        identical.
    Remarks     : an enzyme with a site CGNNCG is palindromic even if some
                  of the sites that it will recognise are not.
                  for example here : CGAACG

    Internal use only. Not meant to be instantiated.
    """

    @classmethod
    def _search(cls):
        """Return a list of cutting sites of the enzyme in the sequence (PRIVATE).

        For internal use only.

        Implement the search method for palindromic enzymes.
        """
        siteloc = cls.dna.finditer(cls.compsite, cls.size)
        cls.results = [r for s, g in siteloc for r in cls._modify(s)]
        if cls.results:
            cls._drop()
        return cls.results

    @classmethod
    def is_palindromic(cls):
        """Return if the enzyme has a palindromic recoginition site."""
        return True