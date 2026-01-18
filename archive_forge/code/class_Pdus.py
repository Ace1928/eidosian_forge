from pyasn1.type import namedtype
from pyasn1.type import namedval
from pyasn1.type import tag
from pyasn1.type import univ
from pyasn1_modules import rfc1155
class Pdus(univ.Choice):
    componentType = namedtype.NamedTypes(namedtype.NamedType('get-request', GetRequestPDU()), namedtype.NamedType('get-next-request', GetNextRequestPDU()), namedtype.NamedType('get-response', GetResponsePDU()), namedtype.NamedType('set-request', SetRequestPDU()), namedtype.NamedType('trap', TrapPDU()))