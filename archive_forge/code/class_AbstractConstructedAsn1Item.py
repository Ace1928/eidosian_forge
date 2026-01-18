import sys
from pyasn1 import error
from pyasn1.compat import calling
from pyasn1.type import constraint
from pyasn1.type import tag
from pyasn1.type import tagmap
class AbstractConstructedAsn1Item(Asn1ItemBase):
    strictConstraints = False
    componentType = None
    sizeSpec = None

    def __init__(self, **kwargs):
        readOnly = {'componentType': self.componentType, 'sizeSpec': self.sizeSpec}
        readOnly.update(kwargs)
        Asn1ItemBase.__init__(self, **readOnly)
        self._componentValues = []

    def __repr__(self):
        representation = '%s %s object at 0x%x' % (self.__class__.__name__, self.isValue and 'value' or 'schema', id(self))
        for attr, value in self.readOnly.items():
            if value is not noValue:
                representation += ' %s=%r' % (attr, value)
        if self.isValue and self._componentValues:
            representation += ' payload [%s]' % ', '.join([repr(x) for x in self._componentValues])
        return '<%s>' % representation

    def __eq__(self, other):
        return self is other and True or self._componentValues == other

    def __ne__(self, other):
        return self._componentValues != other

    def __lt__(self, other):
        return self._componentValues < other

    def __le__(self, other):
        return self._componentValues <= other

    def __gt__(self, other):
        return self._componentValues > other

    def __ge__(self, other):
        return self._componentValues >= other
    if sys.version_info[0] <= 2:

        def __nonzero__(self):
            return self._componentValues and True or False
    else:

        def __bool__(self):
            return self._componentValues and True or False

    def __len__(self):
        return len(self._componentValues)

    def _cloneComponentValues(self, myClone, cloneValueFlag):
        pass

    def clone(self, **kwargs):
        """Create a modified version of |ASN.1| schema object.

        The `clone()` method accepts the same set arguments as |ASN.1|
        class takes on instantiation except that all arguments
        of the `clone()` method are optional.

        Whatever arguments are supplied, they are used to create a copy
        of `self` taking precedence over the ones used to instantiate `self`.

        Possible values of `self` are never copied over thus `clone()` can
        only create a new schema object.

        Returns
        -------
        :
            new instance of |ASN.1| type/value

        Note
        ----
        Due to the mutable nature of the |ASN.1| object, even if no arguments
        are supplied, new |ASN.1| object will always be created as a shallow
        copy of `self`.
        """
        cloneValueFlag = kwargs.pop('cloneValueFlag', False)
        initilaizers = self.readOnly.copy()
        initilaizers.update(kwargs)
        clone = self.__class__(**initilaizers)
        if cloneValueFlag:
            self._cloneComponentValues(clone, cloneValueFlag)
        return clone

    def subtype(self, **kwargs):
        """Create a specialization of |ASN.1| schema object.

        The `subtype()` method accepts the same set arguments as |ASN.1|
        class takes on instantiation except that all parameters
        of the `subtype()` method are optional.

        With the exception of the arguments described below, the rest of
        supplied arguments they are used to create a copy of `self` taking
        precedence over the ones used to instantiate `self`.

        The following arguments to `subtype()` create a ASN.1 subtype out of
        |ASN.1| type.

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
            new instance of |ASN.1| type/value

        Note
        ----
        Due to the immutable nature of the |ASN.1| object, if no arguments
        are supplied, no new |ASN.1| object will be created and `self` will
        be returned instead.
        """
        initializers = self.readOnly.copy()
        cloneValueFlag = kwargs.pop('cloneValueFlag', False)
        implicitTag = kwargs.pop('implicitTag', None)
        if implicitTag is not None:
            initializers['tagSet'] = self.tagSet.tagImplicitly(implicitTag)
        explicitTag = kwargs.pop('explicitTag', None)
        if explicitTag is not None:
            initializers['tagSet'] = self.tagSet.tagExplicitly(explicitTag)
        for arg, option in kwargs.items():
            initializers[arg] += option
        clone = self.__class__(**initializers)
        if cloneValueFlag:
            self._cloneComponentValues(clone, cloneValueFlag)
        return clone

    def verifySizeSpec(self):
        self.sizeSpec(self)

    def getComponentByPosition(self, idx):
        raise error.PyAsn1Error('Method not implemented')

    def setComponentByPosition(self, idx, value, verifyConstraints=True):
        raise error.PyAsn1Error('Method not implemented')

    def setComponents(self, *args, **kwargs):
        for idx, value in enumerate(args):
            self[idx] = value
        for k in kwargs:
            self[k] = kwargs[k]
        return self

    def clear(self):
        self._componentValues = []

    def setDefaultComponents(self):
        pass

    def getComponentType(self):
        return self.componentType