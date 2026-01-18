from pyasn1.type import namedtype
from pyasn1.type import univ
from pyasn1_modules import rfc5035
from pyasn1_modules import rfc5652
class MultipleSignatures(univ.Sequence):
    componentType = namedtype.NamedTypes(namedtype.NamedType('bodyHashAlg', rfc5652.DigestAlgorithmIdentifier()), namedtype.NamedType('signAlg', rfc5652.SignatureAlgorithmIdentifier()), namedtype.NamedType('signAttrsHash', SignAttrsHash()), namedtype.OptionalNamedType('cert', rfc5035.ESSCertIDv2()))