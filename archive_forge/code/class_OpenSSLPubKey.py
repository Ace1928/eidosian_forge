from pyasn1.type import univ, namedtype, tag
class OpenSSLPubKey(univ.Sequence):
    componentType = namedtype.NamedTypes(namedtype.NamedType('header', PubKeyHeader()), namedtype.NamedType('key', univ.OctetString().subtype(implicitTag=tag.Tag(tagClass=0, tagFormat=0, tagId=3))))