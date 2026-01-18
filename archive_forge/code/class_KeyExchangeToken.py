import saml2
from saml2 import SamlBase
from saml2.ws import wsaddr as wsa
from saml2.ws import wssec as wsse
from saml2.ws import wsutil as wsu
class KeyExchangeToken(KeyExchangeTokenType_):
    """The http://docs.oasis-open.org/ws-sx/ws-trust/200512/:KeyExchangeToken element"""
    c_tag = 'KeyExchangeToken'
    c_namespace = NAMESPACE
    c_children = KeyExchangeTokenType_.c_children.copy()
    c_attributes = KeyExchangeTokenType_.c_attributes.copy()
    c_child_order = KeyExchangeTokenType_.c_child_order[:]
    c_cardinality = KeyExchangeTokenType_.c_cardinality.copy()