import saml2
from saml2 import SamlBase
from saml2.ws import wsaddr as wsa
from saml2.ws import wssec as wsse
from saml2.ws import wsutil as wsu
class Claims(ClaimsType_):
    """The http://docs.oasis-open.org/ws-sx/ws-trust/200512/:Claims element"""
    c_tag = 'Claims'
    c_namespace = NAMESPACE
    c_children = ClaimsType_.c_children.copy()
    c_attributes = ClaimsType_.c_attributes.copy()
    c_child_order = ClaimsType_.c_child_order[:]
    c_cardinality = ClaimsType_.c_cardinality.copy()