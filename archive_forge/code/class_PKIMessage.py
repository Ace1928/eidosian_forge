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
class PKIMessage(univ.Sequence):
    """
    PKIMessage ::= SEQUENCE {
    header           PKIHeader,
    body             PKIBody,
    protection   [0] PKIProtection OPTIONAL,
    extraCerts   [1] SEQUENCE SIZE (1..MAX) OF CMPCertificate
                  OPTIONAL
     }"""
    componentType = namedtype.NamedTypes(namedtype.NamedType('header', PKIHeader()), namedtype.NamedType('body', PKIBody()), namedtype.OptionalNamedType('protection', PKIProtection().subtype(explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 0))), namedtype.OptionalNamedType('extraCerts', univ.SequenceOf(componentType=CMPCertificate()).subtype(subtypeSpec=constraint.ValueSizeConstraint(1, MAX), explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 1))))