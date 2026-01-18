from pyasn1.type import constraint
from pyasn1.type import namedtype
from pyasn1.type import namedval
from pyasn1.type import tag
from pyasn1.type import useful
from pyasn1.type import univ
from pyasn1_modules import rfc5280
class SignatureValidationPolicy(univ.Sequence):
    componentType = namedtype.NamedTypes(namedtype.NamedType('signingPeriod', SigningPeriod()), namedtype.NamedType('commonRules', CommonRules()), namedtype.NamedType('commitmentRules', CommitmentRules()), namedtype.OptionalNamedType('signPolExtensions', SignPolExtensions()))