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
class Boolean(Integer):
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
        Object representing non-default ASN.1 subtype constraint(s).Constraints
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

        class RoundResult(Boolean):
            '''
            ASN.1 specification:

            RoundResult ::= BOOLEAN

            ok RoundResult ::= TRUE
            ko RoundResult ::= FALSE
            '''
        ok = RoundResult(True)
        ko = RoundResult(False)
    """
    tagSet = tag.initTagSet(tag.Tag(tag.tagClassUniversal, tag.tagFormatSimple, 1))
    subtypeSpec = Integer.subtypeSpec + constraint.SingleValueConstraint(0, 1)
    namedValues = namedval.NamedValues(('False', 0), ('True', 1))
    typeId = Integer.getTypeId()