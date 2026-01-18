from pyasn1.type import char
from pyasn1.type import constraint
from pyasn1.type import namedtype
from pyasn1.type import namedval
from pyasn1.type import tag
from pyasn1.type import univ
from pyasn1.type import useful
from pyasn1_modules import rfc2314
from pyasn1_modules import rfc2459
from pyasn1_modules import rfc2511
class CertRepMessage(univ.Sequence):
    """
    CertRepMessage ::= SEQUENCE {
         caPubs       [1] SEQUENCE SIZE (1..MAX) OF CMPCertificate
                          OPTIONAL,
         response         SEQUENCE OF CertResponse
     }
    """
    componentType = namedtype.NamedTypes(namedtype.OptionalNamedType('caPubs', univ.SequenceOf(componentType=CMPCertificate()).subtype(subtypeSpec=constraint.ValueSizeConstraint(1, MAX), explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 1))), namedtype.NamedType('response', univ.SequenceOf(componentType=CertResponse())))