from pyasn1.type import constraint
from pyasn1.type import namedtype
from pyasn1.type import namedval
from pyasn1.type import tag
from pyasn1.type import univ
class AttributeTypeAndValues(univ.Sequence):
    componentType = namedtype.NamedTypes(namedtype.NamedType('type', AttributeDescription()), namedtype.NamedType('vals', univ.SetOf(componentType=AttributeValue())))