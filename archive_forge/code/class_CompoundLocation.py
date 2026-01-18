import functools
import re
import warnings
from abc import ABC, abstractmethod
from Bio import BiopythonDeprecationWarning
from Bio import BiopythonParserWarning
from Bio.Seq import MutableSeq
from Bio.Seq import reverse_complement
from Bio.Seq import Seq
class CompoundLocation(Location):
    """For handling joins etc where a feature location has several parts."""

    def __init__(self, parts, operator='join'):
        """Initialize the class.

        >>> from Bio.SeqFeature import SimpleLocation, CompoundLocation
        >>> f1 = SimpleLocation(10, 40, strand=+1)
        >>> f2 = SimpleLocation(50, 59, strand=+1)
        >>> f = CompoundLocation([f1, f2])
        >>> len(f) == len(f1) + len(f2) == 39 == len(list(f))
        True
        >>> print(f.operator)
        join
        >>> 5 in f
        False
        >>> 15 in f
        True
        >>> f.strand
        1

        Notice that the strand of the compound location is computed
        automatically - in the case of mixed strands on the sub-locations
        the overall strand is set to None.

        >>> f = CompoundLocation([SimpleLocation(3, 6, strand=+1),
        ...                       SimpleLocation(10, 13, strand=-1)])
        >>> print(f.strand)
        None
        >>> len(f)
        6
        >>> list(f)
        [3, 4, 5, 12, 11, 10]

        The example above doing list(f) iterates over the coordinates within the
        feature. This allows you to use max and min on the location, to find the
        range covered:

        >>> min(f)
        3
        >>> max(f)
        12

        More generally, you can use the compound location's start and end which
        give the full span covered, 0 <= start <= end <= full sequence length.

        >>> f.start == min(f)
        True
        >>> f.end == max(f) + 1
        True

        This is consistent with the behavior of the SimpleLocation for a single
        region, where again the 'start' and 'end' do not necessarily give the
        biological start and end, but rather the 'minimal' and 'maximal'
        coordinate boundaries.

        Note that adding locations provides a more intuitive method of
        construction:

        >>> f = SimpleLocation(3, 6, strand=+1) + SimpleLocation(10, 13, strand=-1)
        >>> len(f)
        6
        >>> list(f)
        [3, 4, 5, 12, 11, 10]
        """
        self.operator = operator
        self.parts = list(parts)
        for loc in self.parts:
            if not isinstance(loc, SimpleLocation):
                raise ValueError('CompoundLocation should be given a list of SimpleLocation objects, not %s' % loc.__class__)
        if len(parts) < 2:
            raise ValueError(f'CompoundLocation should have at least 2 parts, not {parts!r}')

    def __str__(self):
        """Return a representation of the CompoundLocation object (with python counting)."""
        return '%s{%s}' % (self.operator, ', '.join((str(loc) for loc in self.parts)))

    def __repr__(self):
        """Represent the CompoundLocation object as string for debugging."""
        return f'{self.__class__.__name__}({self.parts!r}, {self.operator!r})'

    def _get_strand(self):
        """Get function for the strand property (PRIVATE)."""
        if len({loc.strand for loc in self.parts}) == 1:
            return self.parts[0].strand
        else:
            return None

    def _set_strand(self, value):
        """Set function for the strand property (PRIVATE)."""
        for loc in self.parts:
            loc.strand = value
    strand = property(fget=_get_strand, fset=_set_strand, doc='Overall strand of the compound location.\n\n        If all the parts have the same strand, that is returned. Otherwise\n        for mixed strands, this returns None.\n\n        >>> from Bio.SeqFeature import SimpleLocation, CompoundLocation\n        >>> f1 = SimpleLocation(15, 17, strand=1)\n        >>> f2 = SimpleLocation(20, 30, strand=-1)\n        >>> f = f1 + f2\n        >>> f1.strand\n        1\n        >>> f2.strand\n        -1\n        >>> f.strand\n        >>> f.strand is None\n        True\n\n        If you set the strand of a CompoundLocation, this is applied to\n        all the parts - use with caution:\n\n        >>> f.strand = 1\n        >>> f1.strand\n        1\n        >>> f2.strand\n        1\n        >>> f.strand\n        1\n\n        ')

    def __add__(self, other):
        """Combine locations, or shift the location by an integer offset.

        >>> from Bio.SeqFeature import SimpleLocation
        >>> f1 = SimpleLocation(15, 17) + SimpleLocation(20, 30)
        >>> print(f1)
        join{[15:17], [20:30]}

        You can add another SimpleLocation:

        >>> print(f1 + SimpleLocation(40, 50))
        join{[15:17], [20:30], [40:50]}
        >>> print(SimpleLocation(5, 10) + f1)
        join{[5:10], [15:17], [20:30]}

        You can also add another CompoundLocation:

        >>> f2 = SimpleLocation(40, 50) + SimpleLocation(60, 70)
        >>> print(f2)
        join{[40:50], [60:70]}
        >>> print(f1 + f2)
        join{[15:17], [20:30], [40:50], [60:70]}

        Also, as with the SimpleLocation, adding an integer shifts the
        location's coordinates by that offset:

        >>> print(f1 + 100)
        join{[115:117], [120:130]}
        >>> print(200 + f1)
        join{[215:217], [220:230]}
        >>> print(f1 + (-5))
        join{[10:12], [15:25]}
        """
        if isinstance(other, SimpleLocation):
            return CompoundLocation(self.parts + [other], self.operator)
        elif isinstance(other, CompoundLocation):
            if self.operator != other.operator:
                raise ValueError(f'Mixed operators {self.operator} and {other.operator}')
            return CompoundLocation(self.parts + other.parts, self.operator)
        elif isinstance(other, int):
            return self._shift(other)
        else:
            raise NotImplementedError

    def __radd__(self, other):
        """Add a feature to the left."""
        if isinstance(other, SimpleLocation):
            return CompoundLocation([other] + self.parts, self.operator)
        elif isinstance(other, int):
            return self._shift(other)
        else:
            raise NotImplementedError

    def __contains__(self, value):
        """Check if an integer position is within the CompoundLocation object."""
        for loc in self.parts:
            if value in loc:
                return True
        return False

    def __nonzero__(self):
        """Return True regardless of the length of the feature.

        This behavior is for backwards compatibility, since until the
        __len__ method was added, a SimpleLocation always evaluated as True.

        Note that in comparison, Seq objects, strings, lists, etc, will all
        evaluate to False if they have length zero.

        WARNING: The SimpleLocation may in future evaluate to False when its
        length is zero (in order to better match normal python behavior)!
        """
        return True

    def __len__(self):
        """Return the length of the CompoundLocation object."""
        return sum((len(loc) for loc in self.parts))

    def __iter__(self):
        """Iterate over the parent positions within the CompoundLocation object."""
        for loc in self.parts:
            yield from loc

    def __eq__(self, other):
        """Check if all parts of CompoundLocation are equal to all parts of other CompoundLocation."""
        if not isinstance(other, CompoundLocation):
            return False
        if len(self.parts) != len(other.parts):
            return False
        if self.operator != other.operator:
            return False
        for self_part, other_part in zip(self.parts, other.parts):
            if self_part != other_part:
                return False
        return True

    def _shift(self, offset):
        """Return a copy of the CompoundLocation shifted by an offset (PRIVATE)."""
        return CompoundLocation([loc._shift(offset) for loc in self.parts], self.operator)

    def _flip(self, length):
        """Return a copy of the locations after the parent is reversed (PRIVATE).

        Note that the order of the parts is NOT reversed too. Consider a CDS
        on the forward strand with exons small, medium and large (in length).
        Once we change the frame of reference to the reverse complement strand,
        the start codon is still part of the small exon, and the stop codon
        still part of the large exon - so the part order remains the same!

        Here is an artificial example, were the features map to the two upper
        case regions and the lower case runs of n are not used:

        >>> from Bio.Seq import Seq
        >>> from Bio.SeqFeature import SimpleLocation
        >>> dna = Seq("nnnnnAGCATCCTGCTGTACnnnnnnnnGAGAMTGCCATGCCCCTGGAGTGAnnnnn")
        >>> small = SimpleLocation(5, 20, strand=1)
        >>> large = SimpleLocation(28, 52, strand=1)
        >>> location = small + large
        >>> print(small)
        [5:20](+)
        >>> print(large)
        [28:52](+)
        >>> print(location)
        join{[5:20](+), [28:52](+)}
        >>> for part in location.parts:
        ...     print(len(part))
        ...
        15
        24

        As you can see, this is a silly example where each "exon" is a word:

        >>> print(small.extract(dna).translate())
        SILLY
        >>> print(large.extract(dna).translate())
        EXAMPLE*
        >>> print(location.extract(dna).translate())
        SILLYEXAMPLE*
        >>> for part in location.parts:
        ...     print(part.extract(dna).translate())
        ...
        SILLY
        EXAMPLE*

        Now, let's look at this from the reverse strand frame of reference:

        >>> flipped_dna = dna.reverse_complement()
        >>> flipped_location = location._flip(len(dna))
        >>> print(flipped_location.extract(flipped_dna).translate())
        SILLYEXAMPLE*
        >>> for part in flipped_location.parts:
        ...     print(part.extract(flipped_dna).translate())
        ...
        SILLY
        EXAMPLE*

        The key point here is the first part of the CompoundFeature is still the
        small exon, while the second part is still the large exon:

        >>> for part in flipped_location.parts:
        ...     print(len(part))
        ...
        15
        24
        >>> print(flipped_location)
        join{[37:52](-), [5:29](-)}

        Notice the parts are not reversed. However, there was a bug here in older
        versions of Biopython which would have given join{[5:29](-), [37:52](-)}
        and the translation would have wrongly been "EXAMPLE*SILLY" instead.

        """
        return CompoundLocation([loc._flip(length) for loc in self.parts], self.operator)

    @property
    def start(self):
        """Start location - left most (minimum) value, regardless of strand.

        Read only, returns an integer like position object, possibly a fuzzy
        position.

        For the special case of a CompoundLocation wrapping the origin of a
        circular genome, this will return zero.
        """
        return min((loc.start for loc in self.parts))

    @property
    def end(self):
        """End location - right most (maximum) value, regardless of strand.

        Read only, returns an integer like position object, possibly a fuzzy
        position.

        For the special case of a CompoundLocation wrapping the origin of
        a circular genome this will match the genome length.
        """
        return max((loc.end for loc in self.parts))

    @property
    def ref(self):
        """Not present in CompoundLocation, dummy method for API compatibility."""
        return None

    @property
    def ref_db(self):
        """Not present in CompoundLocation, dummy method for API compatibility."""
        return None

    def extract(self, parent_sequence, references=None):
        """Extract the sequence from supplied parent sequence using the CompoundLocation object.

        The parent_sequence can be a Seq like object or a string, and will
        generally return an object of the same type. The exception to this is
        a MutableSeq as the parent sequence will return a Seq object.
        If the location refers to other records, they must be supplied
        in the optional dictionary references.

        >>> from Bio.Seq import Seq
        >>> from Bio.SeqFeature import SimpleLocation, CompoundLocation
        >>> seq = Seq("MKQHKAMIVALIVICITAVVAAL")
        >>> fl1 = SimpleLocation(2, 8)
        >>> fl2 = SimpleLocation(10, 15)
        >>> fl3 = CompoundLocation([fl1,fl2])
        >>> fl3.extract(seq)
        Seq('QHKAMILIVIC')

        """
        parts = [loc.extract(parent_sequence, references=references) for loc in self.parts]
        f_seq = functools.reduce(lambda x, y: x + y, parts)
        return f_seq