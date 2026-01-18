from pyasn1.type import namedtype
from pyasn1.type import namedval
from pyasn1.type import tag
from pyasn1.type import univ
from pyasn1.type import useful
from pyasn1_modules import rfc2459
class TBSRequest(univ.Sequence):
    componentType = namedtype.NamedTypes(namedtype.DefaultedNamedType('version', Version('v1').subtype(explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 0))), namedtype.OptionalNamedType('requestorName', GeneralName().subtype(explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 1))), namedtype.NamedType('requestList', univ.SequenceOf(componentType=Request())), namedtype.OptionalNamedType('requestExtensions', rfc2459.Extensions().subtype(explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 2))))