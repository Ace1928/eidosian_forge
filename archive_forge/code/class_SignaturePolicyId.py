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
class SignaturePolicyId(univ.Sequence):
    componentType = namedtype.NamedTypes(namedtype.NamedType('sigPolicyId', SigPolicyId()), namedtype.NamedType('sigPolicyHash', SigPolicyHash()), namedtype.OptionalNamedType('sigPolicyQualifiers', univ.SequenceOf(componentType=SigPolicyQualifierInfo()).subtype(subtypeSpec=constraint.ValueSizeConstraint(1, MAX))))