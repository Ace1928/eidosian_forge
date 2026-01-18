import sys
from pyasn1 import error
from pyasn1.compat import calling
from pyasn1.type import constraint
from pyasn1.type import tag
from pyasn1.type import tagmap
class Asn1ItemBase(Asn1Item):
    tagSet = tag.TagSet()
    subtypeSpec = constraint.ConstraintsIntersection()
    typeId = None

    def __init__(self, **kwargs):
        readOnly = {'tagSet': self.tagSet, 'subtypeSpec': self.subtypeSpec}
        readOnly.update(kwargs)
        self.__dict__.update(readOnly)
        self._readOnly = readOnly

    def __setattr__(self, name, value):
        if name[0] != '_' and name in self._readOnly:
            raise error.PyAsn1Error('read-only instance attribute "%s"' % name)
        self.__dict__[name] = value

    def __str__(self):
        return self.prettyPrint()

    @property
    def readOnly(self):
        return self._readOnly

    @property
    def effectiveTagSet(self):
        """For |ASN.1| type is equivalent to *tagSet*
        """
        return self.tagSet

    @property
    def tagMap(self):
        """Return a :class:`~pyasn1.type.tagmap.TagMap` object mapping ASN.1 tags to ASN.1 objects within callee object.
        """
        return tagmap.TagMap({self.tagSet: self})

    def isSameTypeWith(self, other, matchTags=True, matchConstraints=True):
        """Examine |ASN.1| type for equality with other ASN.1 type.

        ASN.1 tags (:py:mod:`~pyasn1.type.tag`) and constraints
        (:py:mod:`~pyasn1.type.constraint`) are examined when carrying
        out ASN.1 types comparison.

        Python class inheritance relationship is NOT considered.

        Parameters
        ----------
        other: a pyasn1 type object
            Class instance representing ASN.1 type.

        Returns
        -------
        : :class:`bool`
            :class:`True` if *other* is |ASN.1| type,
            :class:`False` otherwise.
        """
        return self is other or ((not matchTags or self.tagSet == other.tagSet) and (not matchConstraints or self.subtypeSpec == other.subtypeSpec))

    def isSuperTypeOf(self, other, matchTags=True, matchConstraints=True):
        """Examine |ASN.1| type for subtype relationship with other ASN.1 type.

        ASN.1 tags (:py:mod:`~pyasn1.type.tag`) and constraints
        (:py:mod:`~pyasn1.type.constraint`) are examined when carrying
        out ASN.1 types comparison.

        Python class inheritance relationship is NOT considered.

        Parameters
        ----------
            other: a pyasn1 type object
                Class instance representing ASN.1 type.

        Returns
        -------
            : :class:`bool`
                :class:`True` if *other* is a subtype of |ASN.1| type,
                :class:`False` otherwise.
        """
        return not matchTags or (self.tagSet.isSuperTagSetOf(other.tagSet) and (not matchConstraints or self.subtypeSpec.isSuperTypeOf(other.subtypeSpec)))

    @staticmethod
    def isNoValue(*values):
        for value in values:
            if value is not noValue:
                return False
        return True

    def prettyPrint(self, scope=0):
        raise NotImplementedError()

    def getTagSet(self):
        return self.tagSet

    def getEffectiveTagSet(self):
        return self.effectiveTagSet

    def getTagMap(self):
        return self.tagMap

    def getSubtypeSpec(self):
        return self.subtypeSpec

    def hasValue(self):
        return self.isValue