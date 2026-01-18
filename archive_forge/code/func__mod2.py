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
def _mod2(cls, other):
    """Test if other enzyme produces compatible ends for enzyme (PRIVATE).

        For internal use only.

        Test for the compatibility of restriction ending of RE and other.
        """
    raise ValueError('%s.mod2(%s), %s : NotDefined. pas glop pas glop!' % (str(cls), str(other), str(cls)))