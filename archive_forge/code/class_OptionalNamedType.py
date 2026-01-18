import sys
from pyasn1 import error
from pyasn1.type import tag
from pyasn1.type import tagmap
class OptionalNamedType(NamedType):
    __doc__ = NamedType.__doc__
    isOptional = True