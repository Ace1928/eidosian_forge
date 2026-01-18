import saml2
from saml2 import SamlBase
from saml2.ws import wsaddr as wsa
from saml2.ws import wssec as wsse
from saml2.ws import wsutil as wsu
class IssuedTokens(RequestSecurityTokenResponseCollectionType_):
    """The http://docs.oasis-open.org/ws-sx/ws-trust/200512/:IssuedTokens element"""
    c_tag = 'IssuedTokens'
    c_namespace = NAMESPACE
    c_children = RequestSecurityTokenResponseCollectionType_.c_children.copy()
    c_attributes = RequestSecurityTokenResponseCollectionType_.c_attributes.copy()
    c_child_order = RequestSecurityTokenResponseCollectionType_.c_child_order[:]
    c_cardinality = RequestSecurityTokenResponseCollectionType_.c_cardinality.copy()