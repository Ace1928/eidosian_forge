import saml2
from saml2 import SamlBase
from saml2.ws import wsaddr as wsa
from saml2.ws import wssec as wsse
from saml2.ws import wsutil as wsu
class UseKey(UseKeyType_):
    """The http://docs.oasis-open.org/ws-sx/ws-trust/200512/:UseKey element"""
    c_tag = 'UseKey'
    c_namespace = NAMESPACE
    c_children = UseKeyType_.c_children.copy()
    c_attributes = UseKeyType_.c_attributes.copy()
    c_child_order = UseKeyType_.c_child_order[:]
    c_cardinality = UseKeyType_.c_cardinality.copy()