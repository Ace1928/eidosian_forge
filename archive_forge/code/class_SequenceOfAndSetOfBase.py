import math
import sys
from pyasn1 import error
from pyasn1.codec.ber import eoo
from pyasn1.compat import integer
from pyasn1.compat import octets
from pyasn1.type import base
from pyasn1.type import constraint
from pyasn1.type import namedtype
from pyasn1.type import namedval
from pyasn1.type import tag
from pyasn1.type import tagmap
class SequenceOfAndSetOfBase(base.ConstructedAsn1Type):
    """Create |ASN.1| schema or value object.

    |ASN.1| class is based on :class:`~pyasn1.type.base.ConstructedAsn1Type`,
    its objects are mutable and duck-type Python :class:`list` objects.

    Keyword Args
    ------------
    componentType : :py:class:`~pyasn1.type.base.PyAsn1Item` derivative
        A pyasn1 object representing ASN.1 type allowed within |ASN.1| type

    tagSet: :py:class:`~pyasn1.type.tag.TagSet`
        Object representing non-default ASN.1 tag(s)

    subtypeSpec: :py:class:`~pyasn1.type.constraint.ConstraintsIntersection`
        Object representing non-default ASN.1 subtype constraint(s). Constraints
        verification for |ASN.1| type can only occur on explicit
        `.isInconsistent` call.

    Examples
    --------

    .. code-block:: python

        class LotteryDraw(SequenceOf):  #  SetOf is similar
            '''
            ASN.1 specification:

            LotteryDraw ::= SEQUENCE OF INTEGER
            '''
            componentType = Integer()

        lotteryDraw = LotteryDraw()
        lotteryDraw.extend([123, 456, 789])
    """

    def __init__(self, *args, **kwargs):
        if args:
            for key, value in zip(('componentType', 'tagSet', 'subtypeSpec'), args):
                if key in kwargs:
                    raise error.PyAsn1Error('Conflicting positional and keyword params!')
                kwargs['componentType'] = value
        self._componentValues = noValue
        base.ConstructedAsn1Type.__init__(self, **kwargs)

    def __getitem__(self, idx):
        try:
            return self.getComponentByPosition(idx)
        except error.PyAsn1Error:
            raise IndexError(sys.exc_info()[1])

    def __setitem__(self, idx, value):
        try:
            self.setComponentByPosition(idx, value)
        except error.PyAsn1Error:
            raise IndexError(sys.exc_info()[1])

    def append(self, value):
        if self._componentValues is noValue:
            pos = 0
        else:
            pos = len(self._componentValues)
        self[pos] = value

    def count(self, value):
        return list(self._componentValues.values()).count(value)

    def extend(self, values):
        for value in values:
            self.append(value)
        if self._componentValues is noValue:
            self._componentValues = {}

    def index(self, value, start=0, stop=None):
        if stop is None:
            stop = len(self)
        indices, values = zip(*self._componentValues.items())
        values = list(values)
        try:
            return indices[values.index(value, start, stop)]
        except error.PyAsn1Error:
            raise ValueError(sys.exc_info()[1])

    def reverse(self):
        self._componentValues.reverse()

    def sort(self, key=None, reverse=False):
        self._componentValues = dict(enumerate(sorted(self._componentValues.values(), key=key, reverse=reverse)))

    def __len__(self):
        if self._componentValues is noValue or not self._componentValues:
            return 0
        return max(self._componentValues) + 1

    def __iter__(self):
        for idx in range(0, len(self)):
            yield self.getComponentByPosition(idx)

    def _cloneComponentValues(self, myClone, cloneValueFlag):
        for idx, componentValue in self._componentValues.items():
            if componentValue is not noValue:
                if isinstance(componentValue, base.ConstructedAsn1Type):
                    myClone.setComponentByPosition(idx, componentValue.clone(cloneValueFlag=cloneValueFlag))
                else:
                    myClone.setComponentByPosition(idx, componentValue.clone())

    def getComponentByPosition(self, idx, default=noValue, instantiate=True):
        """Return |ASN.1| type component value by position.

        Equivalent to Python sequence subscription operation (e.g. `[]`).

        Parameters
        ----------
        idx : :class:`int`
            Component index (zero-based). Must either refer to an existing
            component or to N+1 component (if *componentType* is set). In the latter
            case a new component type gets instantiated and appended to the |ASN.1|
            sequence.

        Keyword Args
        ------------
        default: :class:`object`
            If set and requested component is a schema object, return the `default`
            object instead of the requested component.

        instantiate: :class:`bool`
            If :obj:`True` (default), inner component will be automatically instantiated.
            If :obj:`False` either existing component or the :class:`NoValue` object will be
            returned.

        Returns
        -------
        : :py:class:`~pyasn1.type.base.PyAsn1Item`
            Instantiate |ASN.1| component type or return existing component value

        Examples
        --------

        .. code-block:: python

            # can also be SetOf
            class MySequenceOf(SequenceOf):
                componentType = OctetString()

            s = MySequenceOf()

            # returns component #0 with `.isValue` property False
            s.getComponentByPosition(0)

            # returns None
            s.getComponentByPosition(0, default=None)

            s.clear()

            # returns noValue
            s.getComponentByPosition(0, instantiate=False)

            # sets component #0 to OctetString() ASN.1 schema
            # object and returns it
            s.getComponentByPosition(0, instantiate=True)

            # sets component #0 to ASN.1 value object
            s.setComponentByPosition(0, 'ABCD')

            # returns OctetString('ABCD') value object
            s.getComponentByPosition(0, instantiate=False)

            s.clear()

            # returns noValue
            s.getComponentByPosition(0, instantiate=False)
        """
        if isinstance(idx, slice):
            indices = tuple(range(len(self)))
            return [self.getComponentByPosition(subidx, default, instantiate) for subidx in indices[idx]]
        if idx < 0:
            idx = len(self) + idx
            if idx < 0:
                raise error.PyAsn1Error('SequenceOf/SetOf index is out of range')
        try:
            componentValue = self._componentValues[idx]
        except (KeyError, error.PyAsn1Error):
            if not instantiate:
                return default
            self.setComponentByPosition(idx)
            componentValue = self._componentValues[idx]
        if default is noValue or componentValue.isValue:
            return componentValue
        else:
            return default

    def setComponentByPosition(self, idx, value=noValue, verifyConstraints=True, matchTags=True, matchConstraints=True):
        """Assign |ASN.1| type component by position.

        Equivalent to Python sequence item assignment operation (e.g. `[]`)
        or list.append() (when idx == len(self)).

        Parameters
        ----------
        idx: :class:`int`
            Component index (zero-based). Must either refer to existing
            component or to N+1 component. In the latter case a new component
            type gets instantiated (if *componentType* is set, or given ASN.1
            object is taken otherwise) and appended to the |ASN.1| sequence.

        Keyword Args
        ------------
        value: :class:`object` or :py:class:`~pyasn1.type.base.PyAsn1Item` derivative
            A Python value to initialize |ASN.1| component with (if *componentType* is set)
            or ASN.1 value object to assign to |ASN.1| component.
            If `value` is not given, schema object will be set as a component.

        verifyConstraints: :class:`bool`
             If :obj:`False`, skip constraints validation

        matchTags: :class:`bool`
             If :obj:`False`, skip component tags matching

        matchConstraints: :class:`bool`
             If :obj:`False`, skip component constraints matching

        Returns
        -------
        self

        Raises
        ------
        ~pyasn1.error.ValueConstraintError, ~pyasn1.error.PyAsn1Error
            On constraint violation or bad initializer
        IndexError
            When idx > len(self)
        """
        if isinstance(idx, slice):
            indices = tuple(range(len(self)))
            startIdx = indices and indices[idx][0] or 0
            for subIdx, subValue in enumerate(value):
                self.setComponentByPosition(startIdx + subIdx, subValue, verifyConstraints, matchTags, matchConstraints)
            return self
        if idx < 0:
            idx = len(self) + idx
            if idx < 0:
                raise error.PyAsn1Error('SequenceOf/SetOf index is out of range')
        componentType = self.componentType
        if self._componentValues is noValue:
            componentValues = {}
        else:
            componentValues = self._componentValues
        currentValue = componentValues.get(idx, noValue)
        if value is noValue:
            if componentType is not None:
                value = componentType.clone()
            elif currentValue is noValue:
                raise error.PyAsn1Error('Component type not defined')
        elif not isinstance(value, base.Asn1Item):
            if componentType is not None and isinstance(componentType, base.SimpleAsn1Type):
                value = componentType.clone(value=value)
            elif currentValue is not noValue and isinstance(currentValue, base.SimpleAsn1Type):
                value = currentValue.clone(value=value)
            else:
                raise error.PyAsn1Error('Non-ASN.1 value %r and undefined component type at %r' % (value, self))
        elif componentType is not None and (matchTags or matchConstraints):
            subtypeChecker = self.strictConstraints and componentType.isSameTypeWith or componentType.isSuperTypeOf
            if not subtypeChecker(value, verifyConstraints and matchTags, verifyConstraints and matchConstraints):
                if componentType.typeId != Any.typeId:
                    raise error.PyAsn1Error('Component value is tag-incompatible: %r vs %r' % (value, componentType))
        componentValues[idx] = value
        self._componentValues = componentValues
        return self

    @property
    def componentTagMap(self):
        if self.componentType is not None:
            return self.componentType.tagMap

    @property
    def components(self):
        return [self._componentValues[idx] for idx in sorted(self._componentValues)]

    def clear(self):
        """Remove all components and become an empty |ASN.1| value object.

        Has the same effect on |ASN.1| object as it does on :class:`list`
        built-in.
        """
        self._componentValues = {}
        return self

    def reset(self):
        """Remove all components and become a |ASN.1| schema object.

        See :meth:`isValue` property for more information on the
        distinction between value and schema objects.
        """
        self._componentValues = noValue
        return self

    def prettyPrint(self, scope=0):
        scope += 1
        representation = self.__class__.__name__ + ':\n'
        if not self.isValue:
            return representation
        for idx, componentValue in enumerate(self):
            representation += ' ' * scope
            if componentValue is noValue and self.componentType is not None:
                representation += '<empty>'
            else:
                representation += componentValue.prettyPrint(scope)
        return representation

    def prettyPrintType(self, scope=0):
        scope += 1
        representation = '%s -> %s {\n' % (self.tagSet, self.__class__.__name__)
        if self.componentType is not None:
            representation += ' ' * scope
            representation += self.componentType.prettyPrintType(scope)
        return representation + '\n' + ' ' * (scope - 1) + '}'

    @property
    def isValue(self):
        """Indicate that |ASN.1| object represents ASN.1 value.

        If *isValue* is :obj:`False` then this object represents just ASN.1 schema.

        If *isValue* is :obj:`True` then, in addition to its ASN.1 schema features,
        this object can also be used like a Python built-in object
        (e.g. :class:`int`, :class:`str`, :class:`dict` etc.).

        Returns
        -------
        : :class:`bool`
            :obj:`False` if object represents just ASN.1 schema.
            :obj:`True` if object represents ASN.1 schema and can be used as a normal value.

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
        if self._componentValues is noValue:
            return False
        if len(self._componentValues) != len(self):
            return False
        for componentValue in self._componentValues.values():
            if componentValue is noValue or not componentValue.isValue:
                return False
        return True

    @property
    def isInconsistent(self):
        """Run necessary checks to ensure |ASN.1| object consistency.

        Default action is to verify |ASN.1| object against constraints imposed
        by `subtypeSpec`.

        Raises
        ------
        :py:class:`~pyasn1.error.PyAsn1tError` on any inconsistencies found
        """
        if self.componentType is noValue or not self.subtypeSpec:
            return False
        if self._componentValues is noValue:
            return True
        mapping = {}
        for idx, value in self._componentValues.items():
            if value is noValue:
                continue
            mapping[idx] = value
        try:
            self.subtypeSpec(mapping)
        except error.PyAsn1Error:
            exc = sys.exc_info()[1]
            return exc
        return False