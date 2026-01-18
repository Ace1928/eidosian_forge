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
@classmethod
def _rev_modify(cls, location):
    """Return a generator that moves the cutting position by 1 (PRIVATE).

        for internal use only.

        as _modify for site situated on the antiparallel strand when the
        enzyme is not palindromic
        """
    yield (location - cls.fst3)
    yield (location - cls.scd3)