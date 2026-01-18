from pyasn1.type import char
from pyasn1.type import constraint
from pyasn1.type import namedtype
from pyasn1.type import namedval
from pyasn1.type import univ
from pyasn1_modules import rfc5652
import string
class HeaderField(univ.Sequence):
    componentType = namedtype.NamedTypes(namedtype.NamedType('field-Name', HeaderFieldName()), namedtype.NamedType('field-Value', HeaderFieldValue()), namedtype.DefaultedNamedType('field-Status', HeaderFieldStatus().subtype(value='duplicated')))