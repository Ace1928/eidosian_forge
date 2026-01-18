import saml2
from saml2 import SamlBase
from saml2.ws import wsaddr as wsa
from saml2.ws import wssec as wsse
from saml2.ws import wsutil as wsu
class OnBehalfOf(OnBehalfOfType_):
    """The http://docs.oasis-open.org/ws-sx/ws-trust/200512/:OnBehalfOf element"""
    c_tag = 'OnBehalfOf'
    c_namespace = NAMESPACE
    c_children = OnBehalfOfType_.c_children.copy()
    c_attributes = OnBehalfOfType_.c_attributes.copy()
    c_child_order = OnBehalfOfType_.c_child_order[:]
    c_cardinality = OnBehalfOfType_.c_cardinality.copy()