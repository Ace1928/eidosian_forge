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
class Ov3(AbstractCut):
    """Implement methods for enzymes that produce 3' overhanging ends.

    The enzyme cuts the - strand after the + strand of the DNA.

    Internal use only. Not meant to be instantiated.
    """

    @classmethod
    def catalyse(cls, dna, linear=True):
        """List the sequence fragments after cutting dna with enzyme.

        Return a tuple of dna as will be produced by using RE to restrict the
        dna.

        dna must be a Bio.Seq.Seq instance or a Bio.Seq.MutableSeq instance.

        If linear is False, the sequence is considered to be circular and the
        output will be modified accordingly.
        """
        r = cls.search(dna, linear)
        d = cls.dna
        if not r:
            return (d[1:],)
        fragments = []
        length = len(r) - 1
        if d.is_linear():
            fragments.append(d[1:r[0]])
            if length:
                fragments += [d[r[x]:r[x + 1]] for x in range(length)]
            fragments.append(d[r[-1]:])
        else:
            fragments.append(d[r[-1]:] + d[1:r[0]])
            if not length:
                return tuple(fragments)
            fragments += [d[r[x]:r[x + 1]] for x in range(length)]
        return tuple(fragments)
    catalyze = catalyse

    @classmethod
    def is_blunt(cls):
        """Return if the enzyme produces blunt ends.

        True if the enzyme produces blunt end.

        Related methods:

        - RE.is_3overhang()
        - RE.is_5overhang()
        - RE.is_unknown()

        """
        return False

    @classmethod
    def is_5overhang(cls):
        """Return if the enzymes produces 5' overhanging ends.

        True if the enzyme produces 5' overhang sticky end.

        Related methods:

        - RE.is_3overhang()
        - RE.is_blunt()
        - RE.is_unknown()

        """
        return False

    @classmethod
    def is_3overhang(cls):
        """Return if the enzyme produces 3' overhanging ends.

        True if the enzyme produces 3' overhang sticky end.

        Related methods:

        - RE.is_5overhang()
        - RE.is_blunt()
        - RE.is_unknown()

        """
        return True

    @classmethod
    def overhang(cls):
        """Return the type of the enzyme's overhang as string.

        Can be "3' overhang", "5' overhang", "blunt", "unknown".
        """
        return "3' overhang"

    @classmethod
    def compatible_end(cls, batch=None):
        """List all enzymes that produce compatible ends for the enzyme."""
        if not batch:
            batch = AllEnzymes
        r = sorted((x for x in iter(AllEnzymes) if x.is_3overhang() and x % cls))
        return r

    @classmethod
    def _mod1(cls, other):
        """Test if other enzyme produces compatible ends for enzyme (PRIVATE).

        For internal use only.

        Test for the compatibility of restriction ending of RE and other.
        """
        if issubclass(other, Ov3):
            return cls._mod2(other)
        else:
            return False