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
class AbstractCut(RestrictionType):
    """Implement the methods that are common to all restriction enzymes.

    All the methods are classmethod.

    For internal use only. Not meant to be instantiated.
    """

    @classmethod
    def search(cls, dna, linear=True):
        """Return a list of cutting sites of the enzyme in the sequence.

        Compensate for circular sequences and so on.

        dna must be a Bio.Seq.Seq instance or a Bio.Seq.MutableSeq instance.

        If linear is False, the restriction sites that span over the boundaries
        will be included.

        The positions are the first base of the 3' fragment,
        i.e. the first base after the position the enzyme will cut.
        """
        if isinstance(dna, FormattedSeq):
            cls.dna = dna
            return cls._search()
        else:
            cls.dna = FormattedSeq(dna, linear)
            return cls._search()

    @classmethod
    def all_suppliers(cls):
        """Print all the suppliers of restriction enzyme."""
        supply = sorted((x[0] for x in suppliers_dict.values()))
        print(',\n'.join(supply))

    @classmethod
    def is_equischizomer(cls, other):
        """Test for real isoschizomer.

        True if other is an isoschizomer of RE, but not an neoschizomer,
        else False.

        Equischizomer: same site, same position of restriction.

        >>> from Bio.Restriction import SacI, SstI, SmaI, XmaI
        >>> SacI.is_equischizomer(SstI)
        True
        >>> SmaI.is_equischizomer(XmaI)
        False

        """
        return not cls != other

    @classmethod
    def is_neoschizomer(cls, other):
        """Test for neoschizomer.

        True if other is an isoschizomer of RE, else False.
        Neoschizomer: same site, different position of restriction.
        """
        return cls >> other

    @classmethod
    def is_isoschizomer(cls, other):
        """Test for same recognition site.

        True if other has the same recognition site, else False.

        Isoschizomer: same site.

        >>> from Bio.Restriction import SacI, SstI, SmaI, XmaI
        >>> SacI.is_isoschizomer(SstI)
        True
        >>> SmaI.is_isoschizomer(XmaI)
        True

        """
        return not cls != other or cls >> other

    @classmethod
    def equischizomers(cls, batch=None):
        """List equischizomers of the enzyme.

        Return a tuple of all the isoschizomers of RE.
        If batch is supplied it is used instead of the default AllEnzymes.

        Equischizomer: same site, same position of restriction.
        """
        if not batch:
            batch = AllEnzymes
        r = [x for x in batch if not cls != x]
        i = r.index(cls)
        del r[i]
        r.sort()
        return r

    @classmethod
    def neoschizomers(cls, batch=None):
        """List neoschizomers of the enzyme.

        Return a tuple of all the neoschizomers of RE.
        If batch is supplied it is used instead of the default AllEnzymes.

        Neoschizomer: same site, different position of restriction.
        """
        if not batch:
            batch = AllEnzymes
        r = sorted((x for x in batch if cls >> x))
        return r

    @classmethod
    def isoschizomers(cls, batch=None):
        """List all isoschizomers of the enzyme.

        Return a tuple of all the equischizomers and neoschizomers of RE.
        If batch is supplied it is used instead of the default AllEnzymes.
        """
        if not batch:
            batch = AllEnzymes
        r = [x for x in batch if cls >> x or not cls != x]
        i = r.index(cls)
        del r[i]
        r.sort()
        return r

    @classmethod
    def frequency(cls):
        """Return the theoretically cutting frequency of the enzyme.

        Frequency of the site, given as 'one cut per x bases' (int).
        """
        return cls.freq