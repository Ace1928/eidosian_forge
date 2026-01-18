from pyasn1_modules import rfc2315
from pyasn1_modules.rfc2459 import *
class CertRequest(univ.Sequence):
    componentType = namedtype.NamedTypes(namedtype.NamedType('certReqId', univ.Integer()), namedtype.NamedType('certTemplate', CertTemplate()), namedtype.OptionalNamedType('controls', Controls()))