from pyasn1.type import char
from pyasn1.type import constraint
from pyasn1.type import namedtype
from pyasn1.type import namedval
from pyasn1.type import opentype
from pyasn1.type import tag
from pyasn1.type import univ
from pyasn1.type import useful
class PolicyMapping(univ.Sequence):
    componentType = namedtype.NamedTypes(namedtype.NamedType('issuerDomainPolicy', CertPolicyId()), namedtype.NamedType('subjectDomainPolicy', CertPolicyId()))