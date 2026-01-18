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
class Commercially_available(AbstractCut):
    """Implement methods for enzymes which are commercially available.

    Internal use only. Not meant to be instantiated.
    """

    @classmethod
    def suppliers(cls):
        """Print a list of suppliers of the enzyme."""
        for s in cls.suppl:
            print(suppliers_dict[s][0] + ',')

    @classmethod
    def supplier_list(cls):
        """Return a list of suppliers of the enzyme."""
        return [v[0] for k, v in suppliers_dict.items() if k in cls.suppl]

    @classmethod
    def buffers(cls, supplier):
        """Return the recommended buffer of the supplier for this enzyme.

        Not implemented yet.
        """

    @classmethod
    def is_comm(cls):
        """Return if enzyme is commercially available.

        True if RE has suppliers.
        """
        return True