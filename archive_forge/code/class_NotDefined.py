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
class NotDefined(AbstractCut):
    """Implement methods for enzymes with non-characterized overhangs.

    Correspond to NoCut and Unknown.

    Internal use only. Not meant to be instantiated.
    """

    @classmethod
    def _drop(cls):
        """Remove cuts that are outsite of the sequence (PRIVATE).

        For internal use only.

        Drop the site that are situated outside the sequence in linear
        sequence. Modify the index for site in circular sequences.
        """
        if cls.dna.is_linear():
            return
        else:
            length = len(cls.dna)
            for index, location in enumerate(cls.results):
                if location < 1:
                    cls.results[index] += length
                else:
                    break
            for index, location in enumerate(cls.results[:-1]):
                if location > length:
                    cls.results[-(index + 1)] -= length
                else:
                    break

    @classmethod
    def is_defined(cls):
        """Return if recognition sequence and cut are defined.

        True if the sequence recognised and cut is constant,
        i.e. the recognition site is not degenerated AND the enzyme cut inside
        the site.

        Related methods:

        - RE.is_ambiguous()
        - RE.is_unknown()

        """
        return False

    @classmethod
    def is_ambiguous(cls):
        """Return if recognition sequence and cut may be ambiguous.

        True if the sequence recognised and cut is ambiguous,
        i.e. the recognition site is degenerated AND/OR the enzyme cut outside
        the site.

        Related methods:

        - RE.is_defined()
        - RE.is_unknown()

        """
        return False

    @classmethod
    def is_unknown(cls):
        """Return if recognition sequence is unknown.

        True if the sequence is unknown,
        i.e. the recognition site has not been characterised yet.

        Related methods:

        - RE.is_defined()
        - RE.is_ambiguous()

        """
        return True

    @classmethod
    def _mod2(cls, other):
        """Test if other enzyme produces compatible ends for enzyme (PRIVATE).

        For internal use only.

        Test for the compatibility of restriction ending of RE and other.
        """
        raise ValueError('%s.mod2(%s), %s : NotDefined. pas glop pas glop!' % (str(cls), str(other), str(cls)))

    @classmethod
    def elucidate(cls):
        """Return a string representing the recognition site and cuttings.

        Return a representation of the site with the cut on the (+) strand
        represented as '^' and the cut on the (-) strand as '_'.
        ie:

        >>> from Bio.Restriction import EcoRI, KpnI, EcoRV, SnaI
        >>> EcoRI.elucidate()   # 5' overhang
        'G^AATT_C'
        >>> KpnI.elucidate()    # 3' overhang
        'G_GTAC^C'
        >>> EcoRV.elucidate()   # blunt
        'GAT^_ATC'
        >>> SnaI.elucidate()     # NotDefined, cut profile unknown.
        '? GTATAC ?'
        >>>

        """
        return f'? {cls.site} ?'