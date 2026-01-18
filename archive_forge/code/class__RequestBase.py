from pyasn1.type import namedtype
from pyasn1.type import namedval
from pyasn1.type import tag
from pyasn1.type import univ
from pyasn1_modules import rfc1155
class _RequestBase(univ.Sequence):
    componentType = namedtype.NamedTypes(namedtype.NamedType('request-id', RequestID()), namedtype.NamedType('error-status', ErrorStatus()), namedtype.NamedType('error-index', ErrorIndex()), namedtype.NamedType('variable-bindings', VarBindList()))