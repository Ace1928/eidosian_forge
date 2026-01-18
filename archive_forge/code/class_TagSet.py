from pyasn1 import error
class TagSet(object):
    """Create a collection of ASN.1 tags

    Represents a combination of :class:`~pyasn1.type.tag.Tag` objects
    that can be attached to a ASN.1 type to make types distinguishable
    from each other.

    *TagSet* objects are immutable and duck-type Python :class:`tuple` objects
    holding arbitrary number of :class:`~pyasn1.type.tag.Tag` objects.

    Parameters
    ----------
    baseTag: :class:`~pyasn1.type.tag.Tag`
        Base *Tag* object. This tag survives IMPLICIT tagging.

    *superTags: :class:`~pyasn1.type.tag.Tag`
        Additional *Tag* objects taking part in subtyping.

    Examples
    --------
    .. code-block:: python

        class OrderNumber(NumericString):
            '''
            ASN.1 specification

            Order-number ::=
                [APPLICATION 5] IMPLICIT NumericString
            '''
            tagSet = NumericString.tagSet.tagImplicitly(
                Tag(tagClassApplication, tagFormatSimple, 5)
            )

        orderNumber = OrderNumber('1234')
    """

    def __init__(self, baseTag=(), *superTags):
        self.__baseTag = baseTag
        self.__superTags = superTags
        self.__superTagsClassId = tuple([(superTag.tagClass, superTag.tagId) for superTag in superTags])
        self.__lenOfSuperTags = len(superTags)
        self.__hash = hash(self.__superTagsClassId)

    def __repr__(self):
        representation = '-'.join(['%s:%s:%s' % (x.tagClass, x.tagFormat, x.tagId) for x in self.__superTags])
        if representation:
            representation = 'tags ' + representation
        else:
            representation = 'untagged'
        return '<%s object at 0x%x %s>' % (self.__class__.__name__, id(self), representation)

    def __add__(self, superTag):
        return self.__class__(self.__baseTag, *self.__superTags + (superTag,))

    def __radd__(self, superTag):
        return self.__class__(self.__baseTag, *(superTag,) + self.__superTags)

    def __getitem__(self, i):
        if i.__class__ is slice:
            return self.__class__(self.__baseTag, *self.__superTags[i])
        else:
            return self.__superTags[i]

    def __eq__(self, other):
        return self.__superTagsClassId == other

    def __ne__(self, other):
        return self.__superTagsClassId != other

    def __lt__(self, other):
        return self.__superTagsClassId < other

    def __le__(self, other):
        return self.__superTagsClassId <= other

    def __gt__(self, other):
        return self.__superTagsClassId > other

    def __ge__(self, other):
        return self.__superTagsClassId >= other

    def __hash__(self):
        return self.__hash

    def __len__(self):
        return self.__lenOfSuperTags

    @property
    def baseTag(self):
        """Return base ASN.1 tag

        Returns
        -------
        : :class:`~pyasn1.type.tag.Tag`
            Base tag of this *TagSet*
        """
        return self.__baseTag

    @property
    def superTags(self):
        """Return ASN.1 tags

        Returns
        -------
        : :py:class:`tuple`
            Tuple of :class:`~pyasn1.type.tag.Tag` objects that this *TagSet* contains
        """
        return self.__superTags

    def tagExplicitly(self, superTag):
        """Return explicitly tagged *TagSet*

        Create a new *TagSet* representing callee *TagSet* explicitly tagged
        with passed tag(s). With explicit tagging mode, new tags are appended
        to existing tag(s).

        Parameters
        ----------
        superTag: :class:`~pyasn1.type.tag.Tag`
            *Tag* object to tag this *TagSet*

        Returns
        -------
        : :class:`~pyasn1.type.tag.TagSet`
            New *TagSet* object
        """
        if superTag.tagClass == tagClassUniversal:
            raise error.PyAsn1Error("Can't tag with UNIVERSAL class tag")
        if superTag.tagFormat != tagFormatConstructed:
            superTag = Tag(superTag.tagClass, tagFormatConstructed, superTag.tagId)
        return self + superTag

    def tagImplicitly(self, superTag):
        """Return implicitly tagged *TagSet*

        Create a new *TagSet* representing callee *TagSet* implicitly tagged
        with passed tag(s). With implicit tagging mode, new tag(s) replace the
        last existing tag.

        Parameters
        ----------
        superTag: :class:`~pyasn1.type.tag.Tag`
            *Tag* object to tag this *TagSet*

        Returns
        -------
        : :class:`~pyasn1.type.tag.TagSet`
            New *TagSet* object
        """
        if self.__superTags:
            superTag = Tag(superTag.tagClass, self.__superTags[-1].tagFormat, superTag.tagId)
        return self[:-1] + superTag

    def isSuperTagSetOf(self, tagSet):
        """Test type relationship against given *TagSet*

        The callee is considered to be a supertype of given *TagSet*
        tag-wise if all tags in *TagSet* are present in the callee and
        they are in the same order.

        Parameters
        ----------
        tagSet: :class:`~pyasn1.type.tag.TagSet`
            *TagSet* object to evaluate against the callee

        Returns
        -------
        : :py:class:`bool`
            `True` if callee is a supertype of *tagSet*
        """
        if len(tagSet) < self.__lenOfSuperTags:
            return False
        return self.__superTags == tagSet[:self.__lenOfSuperTags]

    def getBaseTag(self):
        return self.__baseTag