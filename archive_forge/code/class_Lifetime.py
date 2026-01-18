import saml2
from saml2 import SamlBase
from saml2.ws import wsaddr as wsa
from saml2.ws import wssec as wsse
from saml2.ws import wsutil as wsu
class Lifetime(LifetimeType_):
    """The http://docs.oasis-open.org/ws-sx/ws-trust/200512/:Lifetime element"""
    c_tag = 'Lifetime'
    c_namespace = NAMESPACE
    c_children = LifetimeType_.c_children.copy()
    c_attributes = LifetimeType_.c_attributes.copy()
    c_child_order = LifetimeType_.c_child_order[:]
    c_cardinality = LifetimeType_.c_cardinality.copy()