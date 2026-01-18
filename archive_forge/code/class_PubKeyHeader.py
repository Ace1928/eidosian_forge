from pyasn1.type import univ, namedtype, tag
class PubKeyHeader(univ.Sequence):
    componentType = namedtype.NamedTypes(namedtype.NamedType('oid', univ.ObjectIdentifier()), namedtype.NamedType('parameters', univ.Null()))