from pyasn1.type import namedtype
from pyasn1.type import univ
from pyasn1_modules import rfc5280
class ProxyPolicy(univ.Sequence):
    componentType = namedtype.NamedTypes(namedtype.NamedType('policyLanguage', univ.ObjectIdentifier()), namedtype.OptionalNamedType('policy', univ.OctetString()))