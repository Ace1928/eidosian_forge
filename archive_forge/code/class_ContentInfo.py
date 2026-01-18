from pyasn1_modules.rfc2459 import *
class ContentInfo(univ.Sequence):
    componentType = namedtype.NamedTypes(namedtype.NamedType('contentType', ContentType()), namedtype.OptionalNamedType('content', univ.Any().subtype(explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 0)), openType=opentype.OpenType('contentType', contentTypeMap)))