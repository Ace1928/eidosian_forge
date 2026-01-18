import saml2
from saml2 import SamlBase
from saml2.ws import wsaddr as wsa
from saml2.ws import wssec as wsse
from saml2.ws import wsutil as wsu
class RequestType(RequestTypeOpenEnum_):
    """The http://docs.oasis-open.org/ws-sx/ws-trust/200512/:RequestType element"""
    c_tag = 'RequestType'
    c_namespace = NAMESPACE
    c_children = RequestTypeOpenEnum_.c_children.copy()
    c_attributes = RequestTypeOpenEnum_.c_attributes.copy()
    c_child_order = RequestTypeOpenEnum_.c_child_order[:]
    c_cardinality = RequestTypeOpenEnum_.c_cardinality.copy()