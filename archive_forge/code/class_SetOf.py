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
class SetOf(SequenceOfAndSetOfBase):
    __doc__ = SequenceOfAndSetOfBase.__doc__
    tagSet = tag.initTagSet(tag.Tag(tag.tagClassUniversal, tag.tagFormatConstructed, 17))
    componentType = None
    subtypeSpec = constraint.ConstraintsIntersection()
    typeId = SequenceOfAndSetOfBase.getTypeId()