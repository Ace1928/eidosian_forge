import saml2
from saml2 import SamlBase
from saml2.ws import wsaddr as wsa
from saml2.ws import wssec as wsse
from saml2.ws import wsutil as wsu
class RequestedTokenCancelled(RequestedTokenCancelledType_):
    """The http://docs.oasis-open.org/ws-sx/ws-trust/200512/:RequestedTokenCancelled element"""
    c_tag = 'RequestedTokenCancelled'
    c_namespace = NAMESPACE
    c_children = RequestedTokenCancelledType_.c_children.copy()
    c_attributes = RequestedTokenCancelledType_.c_attributes.copy()
    c_child_order = RequestedTokenCancelledType_.c_child_order[:]
    c_cardinality = RequestedTokenCancelledType_.c_cardinality.copy()