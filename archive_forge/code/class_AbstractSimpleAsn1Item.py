import sys
from pyasn1 import error
from pyasn1.compat import calling
from pyasn1.type import constraint
from pyasn1.type import tag
from pyasn1.type import tagmap
class AbstractSimpleAsn1Item(Asn1ItemBase):
    defaultValue = noValue

    def __init__(self, value=noValue, **kwargs):
        Asn1ItemBase.__init__(self, **kwargs)
        if value is noValue:
            value = self.defaultValue
        else:
            value = self.prettyIn(value)
            try:
                self.subtypeSpec(value)
            except error.PyAsn1Error:
                exType, exValue, exTb = sys.exc_info()
                raise exType('%s at %s' % (exValue, self.__class__.__name__))
        self._value = value

    def __repr__(self):
        representation = '%s %s object at 0x%x' % (self.__class__.__name__, self.isValue and 'value' or 'schema', id(self))
        for attr, value in self.readOnly.items():
            if value:
                representation += ' %s %s' % (attr, value)
        if self.isValue:
            value = self.prettyPrint()
            if len(value) > 32:
                value = value[:16] + '...' + value[-16:]
            representation += ' payload [%s]' % value
        return '<%s>' % representation

    def __eq__(self, other):
        return self is other and True or self._value == other

    def __ne__(self, other):
        return self._value != other

    def __lt__(self, other):
        return self._value < other

    def __le__(self, other):
        return self._value <= other

    def __gt__(self, other):
        return self._value > other

    def __ge__(self, other):
        return self._value >= other
    if sys.version_info[0] <= 2:

        def __nonzero__(self):
            return self._value and True or False
    else:

        def __bool__(self):
            return self._value and True or False

    def __hash__(self):
        return hash(self._value)

    @property
    def isValue(self):
        """Indicate that |ASN.1| object represents ASN.1 value.

        If *isValue* is `False` then this object represents just ASN.1 schema.

        If *isValue* is `True` then, in addition to its ASN.1 schema features,
        this object can also be used like a Python built-in object (e.g. `int`,
        `str`, `dict` etc.).

        Returns
        -------
        : :class:`bool`
            :class:`False` if object represents just ASN.1 schema.
            :class:`True` if object represents ASN.1 schema and can be used as a normal value.

        Note
        ----
        There is an important distinction between PyASN1 schema and value objects.
        The PyASN1 schema objects can only participate in ASN.1 schema-related
        operations (e.g. defining or testing the structure of the data). Most
        obvious uses of ASN.1 schema is to guide serialisation codecs whilst
        encoding/decoding serialised ASN.1 contents.

        The PyASN1 value objects can **additionally** participate in many operations
        involving regular Python objects (e.g. arithmetic, comprehension etc).
        """
        return self._value is not noValue

    def clone(self, value=noValue, **kwargs):
        """Create a modified version of |ASN.1| schema or value object.

        The `clone()` method accepts the same set arguments as |ASN.1|
        class takes on instantiation except that all arguments
        of the `clone()` method are optional.

        Whatever arguments are supplied, they are used to create a copy
        of `self` taking precedence over the ones used to instantiate `self`.

        Note
        ----
        Due to the immutable nature of the |ASN.1| object, if no arguments
        are supplied, no new |ASN.1| object will be created and `self` will
        be returned instead.
        """
        if value is noValue:
            if not kwargs:
                return self
            value = self._value
        initilaizers = self.readOnly.copy()
        initilaizers.update(kwargs)
        return self.__class__(value, **initilaizers)

    def subtype(self, value=noValue, **kwargs):
        """Create a specialization of |ASN.1| schema or value object.

        The subtype relationship between ASN.1 types has no correlation with
        subtype relationship between Python types. ASN.1 type is mainly identified
        by its tag(s) (:py:class:`~pyasn1.type.tag.TagSet`) and value range
        constraints (:py:class:`~pyasn1.type.constraint.ConstraintsIntersection`).
        These ASN.1 type properties are implemented as |ASN.1| attributes.  

        The `subtype()` method accepts the same set arguments as |ASN.1|
        class takes on instantiation except that all parameters
        of the `subtype()` method are optional.

        With the exception of the arguments described below, the rest of
        supplied arguments they are used to create a copy of `self` taking
        precedence over the ones used to instantiate `self`.

        The following arguments to `subtype()` create a ASN.1 subtype out of
        |ASN.1| type:

        Other Parameters
        ----------------
        implicitTag: :py:class:`~pyasn1.type.tag.Tag`
            Implicitly apply given ASN.1 tag object to `self`'s
            :py:class:`~pyasn1.type.tag.TagSet`, then use the result as
            new object's ASN.1 tag(s).

        explicitTag: :py:class:`~pyasn1.type.tag.Tag`
            Explicitly apply given ASN.1 tag object to `self`'s
            :py:class:`~pyasn1.type.tag.TagSet`, then use the result as
            new object's ASN.1 tag(s).

        subtypeSpec: :py:class:`~pyasn1.type.constraint.ConstraintsIntersection`
            Add ASN.1 constraints object to one of the `self`'s, then
            use the result as new object's ASN.1 constraints.

        Returns
        -------
        :
            new instance of |ASN.1| schema or value object

        Note
        ----
        Due to the immutable nature of the |ASN.1| object, if no arguments
        are supplied, no new |ASN.1| object will be created and `self` will
        be returned instead.
        """
        if value is noValue:
            if not kwargs:
                return self
            value = self._value
        initializers = self.readOnly.copy()
        implicitTag = kwargs.pop('implicitTag', None)
        if implicitTag is not None:
            initializers['tagSet'] = self.tagSet.tagImplicitly(implicitTag)
        explicitTag = kwargs.pop('explicitTag', None)
        if explicitTag is not None:
            initializers['tagSet'] = self.tagSet.tagExplicitly(explicitTag)
        for arg, option in kwargs.items():
            initializers[arg] += option
        return self.__class__(value, **initializers)

    def prettyIn(self, value):
        return value

    def prettyOut(self, value):
        return str(value)

    def prettyPrint(self, scope=0):
        return self.prettyOut(self._value)

    def prettyPrintType(self, scope=0):
        return '%s -> %s' % (self.tagSet, self.__class__.__name__)