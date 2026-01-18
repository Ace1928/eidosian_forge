from pyasn1.type import constraint
from pyasn1.type import namedtype
from pyasn1.type import namedval
from pyasn1.type import tag
from pyasn1.type import univ
from pyasn1_modules import rfc1902
class VarBind(univ.Sequence):
    componentType = namedtype.NamedTypes(namedtype.NamedType('name', rfc1902.ObjectName()), namedtype.NamedType('', _BindValue()))