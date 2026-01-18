import saml2
from saml2 import SamlBase
from saml2.ws import wsaddr as wsa
from saml2.ws import wssec as wsse
from saml2.ws import wsutil as wsu
class RequestSecurityTokenCollection(RequestSecurityTokenCollectionType_):
    """The http://docs.oasis-open.org/ws-sx/ws-trust/200512/:RequestSecurityTokenCollection element"""
    c_tag = 'RequestSecurityTokenCollection'
    c_namespace = NAMESPACE
    c_children = RequestSecurityTokenCollectionType_.c_children.copy()
    c_attributes = RequestSecurityTokenCollectionType_.c_attributes.copy()
    c_child_order = RequestSecurityTokenCollectionType_.c_child_order[:]
    c_cardinality = RequestSecurityTokenCollectionType_.c_cardinality.copy()