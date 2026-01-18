from pyasn1.type import constraint
from pyasn1.type import namedtype
from pyasn1.type import tag
from pyasn1.type import univ
class ApplicationSyntax(univ.Choice):
    componentType = namedtype.NamedTypes(namedtype.NamedType('address', NetworkAddress()), namedtype.NamedType('counter', Counter()), namedtype.NamedType('gauge', Gauge()), namedtype.NamedType('ticks', TimeTicks()), namedtype.NamedType('arbitrary', Opaque()))