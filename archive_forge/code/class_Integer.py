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
class Integer(base.SimpleAsn1Type):
    """Create |ASN.1| schema or value object.

    |ASN.1| class is based on :class:`~pyasn1.type.base.SimpleAsn1Type`, its
    objects are immutable and duck-type Python :class:`int` objects.

    Keyword Args
    ------------
    value: :class:`int`, :class:`str` or |ASN.1| object
        Python :class:`int` or :class:`str` literal or |ASN.1| class
        instance. If `value` is not given, schema object will be created.

    tagSet: :py:class:`~pyasn1.type.tag.TagSet`
        Object representing non-default ASN.1 tag(s)

    subtypeSpec: :py:class:`~pyasn1.type.constraint.ConstraintsIntersection`
        Object representing non-default ASN.1 subtype constraint(s). Constraints
        verification for |ASN.1| type occurs automatically on object
        instantiation.

    namedValues: :py:class:`~pyasn1.type.namedval.NamedValues`
        Object representing non-default symbolic aliases for numbers

    Raises
    ------
    ~pyasn1.error.ValueConstraintError, ~pyasn1.error.PyAsn1Error
        On constraint violation or bad initializer.

    Examples
    --------

    .. code-block:: python

        class ErrorCode(Integer):
            '''
            ASN.1 specification:

            ErrorCode ::=
                INTEGER { disk-full(1), no-disk(-1),
                          disk-not-formatted(2) }

            error ErrorCode ::= disk-full
            '''
            namedValues = NamedValues(
                ('disk-full', 1), ('no-disk', -1),
                ('disk-not-formatted', 2)
            )

        error = ErrorCode('disk-full')
    """
    tagSet = tag.initTagSet(tag.Tag(tag.tagClassUniversal, tag.tagFormatSimple, 2))
    subtypeSpec = constraint.ConstraintsIntersection()
    namedValues = namedval.NamedValues()
    typeId = base.SimpleAsn1Type.getTypeId()

    def __init__(self, value=noValue, **kwargs):
        if 'namedValues' not in kwargs:
            kwargs['namedValues'] = self.namedValues
        base.SimpleAsn1Type.__init__(self, value, **kwargs)

    def __and__(self, value):
        return self.clone(self._value & value)

    def __rand__(self, value):
        return self.clone(value & self._value)

    def __or__(self, value):
        return self.clone(self._value | value)

    def __ror__(self, value):
        return self.clone(value | self._value)

    def __xor__(self, value):
        return self.clone(self._value ^ value)

    def __rxor__(self, value):
        return self.clone(value ^ self._value)

    def __lshift__(self, value):
        return self.clone(self._value << value)

    def __rshift__(self, value):
        return self.clone(self._value >> value)

    def __add__(self, value):
        return self.clone(self._value + value)

    def __radd__(self, value):
        return self.clone(value + self._value)

    def __sub__(self, value):
        return self.clone(self._value - value)

    def __rsub__(self, value):
        return self.clone(value - self._value)

    def __mul__(self, value):
        return self.clone(self._value * value)

    def __rmul__(self, value):
        return self.clone(value * self._value)

    def __mod__(self, value):
        return self.clone(self._value % value)

    def __rmod__(self, value):
        return self.clone(value % self._value)

    def __pow__(self, value, modulo=None):
        return self.clone(pow(self._value, value, modulo))

    def __rpow__(self, value):
        return self.clone(pow(value, self._value))

    def __floordiv__(self, value):
        return self.clone(self._value // value)

    def __rfloordiv__(self, value):
        return self.clone(value // self._value)
    if sys.version_info[0] <= 2:

        def __div__(self, value):
            if isinstance(value, float):
                return Real(self._value / value)
            else:
                return self.clone(self._value / value)

        def __rdiv__(self, value):
            if isinstance(value, float):
                return Real(value / self._value)
            else:
                return self.clone(value / self._value)
    else:

        def __truediv__(self, value):
            return Real(self._value / value)

        def __rtruediv__(self, value):
            return Real(value / self._value)

        def __divmod__(self, value):
            return self.clone(divmod(self._value, value))

        def __rdivmod__(self, value):
            return self.clone(divmod(value, self._value))
        __hash__ = base.SimpleAsn1Type.__hash__

    def __int__(self):
        return int(self._value)
    if sys.version_info[0] <= 2:

        def __long__(self):
            return long(self._value)

    def __float__(self):
        return float(self._value)

    def __abs__(self):
        return self.clone(abs(self._value))

    def __index__(self):
        return int(self._value)

    def __pos__(self):
        return self.clone(+self._value)

    def __neg__(self):
        return self.clone(-self._value)

    def __invert__(self):
        return self.clone(~self._value)

    def __round__(self, n=0):
        r = round(self._value, n)
        if n:
            return self.clone(r)
        else:
            return r

    def __floor__(self):
        return math.floor(self._value)

    def __ceil__(self):
        return math.ceil(self._value)

    def __trunc__(self):
        return self.clone(math.trunc(self._value))

    def __lt__(self, value):
        return self._value < value

    def __le__(self, value):
        return self._value <= value

    def __eq__(self, value):
        return self._value == value

    def __ne__(self, value):
        return self._value != value

    def __gt__(self, value):
        return self._value > value

    def __ge__(self, value):
        return self._value >= value

    def prettyIn(self, value):
        try:
            return int(value)
        except ValueError:
            try:
                return self.namedValues[value]
            except KeyError:
                raise error.PyAsn1Error("Can't coerce %r into integer: %s" % (value, sys.exc_info()[1]))

    def prettyOut(self, value):
        try:
            return str(self.namedValues[value])
        except KeyError:
            return str(value)

    def getNamedValues(self):
        return self.namedValues