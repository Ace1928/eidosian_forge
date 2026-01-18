import saml2
from saml2 import SamlBase
from saml2.ws import wsaddr as wsa
from saml2.ws import wssec as wsse
from saml2.ws import wsutil as wsu
class RequestSecurityToken(RequestSecurityTokenType_):
    """The http://docs.oasis-open.org/ws-sx/ws-trust/200512/:RequestSecurityToken element"""
    c_tag = 'RequestSecurityToken'
    c_namespace = NAMESPACE
    c_children = RequestSecurityTokenType_.c_children.copy()
    c_attributes = RequestSecurityTokenType_.c_attributes.copy()
    c_child_order = RequestSecurityTokenType_.c_child_order[:]
    c_cardinality = RequestSecurityTokenType_.c_cardinality.copy()