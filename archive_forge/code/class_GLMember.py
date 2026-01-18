from pyasn1.type import constraint
from pyasn1.type import namedtype
from pyasn1.type import namedval
from pyasn1.type import opentype
from pyasn1.type import tag
from pyasn1.type import univ
from pyasn1.type import useful
from pyasn1_modules import rfc3565
from pyasn1_modules import rfc5280
from pyasn1_modules import rfc5652
from pyasn1_modules import rfc5751
from pyasn1_modules import rfc5755
class GLMember(univ.Sequence):
    componentType = namedtype.NamedTypes(namedtype.NamedType('glMemberName', GeneralName()), namedtype.OptionalNamedType('glMemberAddress', GeneralName()), namedtype.OptionalNamedType('certificates', Certificates()))