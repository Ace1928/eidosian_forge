from pyasn1.type import namedtype
from pyasn1.type import namedval
from pyasn1.type import tag
from pyasn1.type import univ
from pyasn1.type import useful
from pyasn1_modules import rfc2459
class OCSPRequest(univ.Sequence):
    componentType = namedtype.NamedTypes(namedtype.NamedType('tbsRequest', TBSRequest()), namedtype.OptionalNamedType('optionalSignature', Signature().subtype(explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 0))))