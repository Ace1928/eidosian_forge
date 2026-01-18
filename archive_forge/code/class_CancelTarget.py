import saml2
from saml2 import SamlBase
from saml2.ws import wsaddr as wsa
from saml2.ws import wssec as wsse
from saml2.ws import wsutil as wsu
class CancelTarget(CancelTargetType_):
    """The http://docs.oasis-open.org/ws-sx/ws-trust/200512/:CancelTarget element"""
    c_tag = 'CancelTarget'
    c_namespace = NAMESPACE
    c_children = CancelTargetType_.c_children.copy()
    c_attributes = CancelTargetType_.c_attributes.copy()
    c_child_order = CancelTargetType_.c_child_order[:]
    c_cardinality = CancelTargetType_.c_cardinality.copy()