from pyasn1.type import constraint
from pyasn1.type import namedtype
from pyasn1.type import tag
from pyasn1.type import univ
from pyasn1_modules import rfc5652
class ROAIPAddress(univ.Sequence):
    componentType = namedtype.NamedTypes(namedtype.NamedType('address', IPAddress()), namedtype.OptionalNamedType('maxLength', univ.Integer()))