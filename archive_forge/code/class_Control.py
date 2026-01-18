from pyasn1.type import constraint
from pyasn1.type import namedtype
from pyasn1.type import namedval
from pyasn1.type import tag
from pyasn1.type import univ
class Control(univ.Sequence):
    componentType = namedtype.NamedTypes(namedtype.NamedType('controlType', LDAPOID()), namedtype.DefaultedNamedType('criticality', univ.Boolean('False')), namedtype.OptionalNamedType('controlValue', univ.OctetString()))