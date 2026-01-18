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
def _test_reverse(self, start, end, site):
    """Test if site is between end and start, for circular sequences (PRIVATE).

        Internal use only.
        """
    return start <= site <= len(self.sequence) or 1 <= site < end