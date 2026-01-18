from pyasn1.type import char
from pyasn1.type import constraint
from pyasn1.type import namedtype
from pyasn1.type import opentype
from pyasn1.type import tag
from pyasn1.type import useful
from pyasn1.type import univ
from pyasn1_modules import rfc5280
from pyasn1_modules import rfc5652
from pyasn1_modules import rfc5035
from pyasn1_modules import rfc5755
from pyasn1_modules import rfc6960
from pyasn1_modules import rfc3161
class OcspIdentifier(univ.Sequence):
    componentType = namedtype.NamedTypes(namedtype.NamedType('ocspResponderID', ResponderID()), namedtype.NamedType('producedAt', useful.GeneralizedTime()))