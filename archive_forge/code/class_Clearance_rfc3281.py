from pyasn1.type import char
from pyasn1.type import constraint
from pyasn1.type import namedtype
from pyasn1.type import namedval
from pyasn1.type import opentype
from pyasn1.type import tag
from pyasn1.type import univ
from pyasn1.type import useful
from pyasn1_modules import rfc5280
from pyasn1_modules import rfc5652
class Clearance_rfc3281(univ.Sequence):
    componentType = namedtype.NamedTypes(namedtype.NamedType('policyId', univ.ObjectIdentifier().subtype(implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 0))), namedtype.DefaultedNamedType('classList', ClassList().subtype(implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 1)).subtype(value='unclassified')), namedtype.OptionalNamedType('securityCategories', univ.SetOf(componentType=SecurityCategory()).subtype(implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 2))))