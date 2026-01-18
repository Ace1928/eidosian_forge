import saml2
from saml2 import SamlBase
from saml2.ws import wsaddr as wsa
from saml2.ws import wssec as wsse
from saml2.ws import wsutil as wsu
class RequestedProofToken(RequestedProofTokenType_):
    """The http://docs.oasis-open.org/ws-sx/ws-trust/200512/:RequestedProofToken element"""
    c_tag = 'RequestedProofToken'
    c_namespace = NAMESPACE
    c_children = RequestedProofTokenType_.c_children.copy()
    c_attributes = RequestedProofTokenType_.c_attributes.copy()
    c_child_order = RequestedProofTokenType_.c_child_order[:]
    c_cardinality = RequestedProofTokenType_.c_cardinality.copy()