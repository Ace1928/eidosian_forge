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
class Not_available(AbstractCut):
    """Implement methods for enzymes which are not commercially available.

    Internal use only. Not meant to be instantiated.
    """

    @staticmethod
    def suppliers():
        """Print a list of suppliers of the enzyme."""
        return None

    @classmethod
    def supplier_list(cls):
        """Return a list of suppliers of the enzyme."""
        return []

    @classmethod
    def buffers(cls, supplier):
        """Return the recommended buffer of the supplier for this enzyme.

        Not implemented yet.
        """
        raise TypeError('Enzyme not commercially available.')

    @classmethod
    def is_comm(cls):
        """Return if enzyme is commercially available.

        True if RE has suppliers.
        """
        return False