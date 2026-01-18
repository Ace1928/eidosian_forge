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
def _sub_set(self, wanted):
    """Filter result for keys which are in wanted (PRIVATE).

        Internal use only. Returns a dict.

        Screen the results through wanted set.
        Keep only the results for which the enzymes is in wanted set.
        """
    return {k: v for k, v in self.mapping.items() if k in wanted}