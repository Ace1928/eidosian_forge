import saml2
from saml2 import SamlBase
from saml2.ws import wsaddr as wsa
from saml2.ws import wssec as wsse
from saml2.ws import wsutil as wsu
class RequestedAttachedReference(RequestedReferenceType_):
    """The http://docs.oasis-open.org/ws-sx/ws-trust/200512/:RequestedAttachedReference element"""
    c_tag = 'RequestedAttachedReference'
    c_namespace = NAMESPACE
    c_children = RequestedReferenceType_.c_children.copy()
    c_attributes = RequestedReferenceType_.c_attributes.copy()
    c_child_order = RequestedReferenceType_.c_child_order[:]
    c_cardinality = RequestedReferenceType_.c_cardinality.copy()