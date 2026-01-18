from pyasn1.type import namedtype
from pyasn1.type import univ
from pyasn1_modules import rfc5280
from pyasn1_modules import rfc4055
class SCVPIssuerSerial(univ.Sequence):
    componentType = namedtype.NamedTypes(namedtype.NamedType('issuer', GeneralNames()), namedtype.NamedType('serialNumber', CertificateSerialNumber()))