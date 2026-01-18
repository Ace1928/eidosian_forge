from pyasn1.type import constraint
from pyasn1.type import univ
from pyasn1_modules import rfc5280
from pyasn1_modules import rfc5755
class AuthorityClearanceConstraints(univ.SequenceOf):
    componentType = rfc5755.Clearance()
    subtypeSpec = constraint.ValueSizeConstraint(1, MAX)