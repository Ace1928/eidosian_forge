import saml2
from saml2 import SamlBase
from saml2.ws import wsaddr as wsa
from saml2.ws import wssec as wsse
from saml2.ws import wsutil as wsu
class ValidateTarget(ValidateTargetType_):
    """The http://docs.oasis-open.org/ws-sx/ws-trust/200512/:ValidateTarget element"""
    c_tag = 'ValidateTarget'
    c_namespace = NAMESPACE
    c_children = ValidateTargetType_.c_children.copy()
    c_attributes = ValidateTargetType_.c_attributes.copy()
    c_child_order = ValidateTargetType_.c_child_order[:]
    c_cardinality = ValidateTargetType_.c_cardinality.copy()