import saml2
from saml2 import SamlBase
from saml2.ws import wsaddr as wsa
from saml2.ws import wssec as wsse
from saml2.ws import wsutil as wsu
class AllowPostdating(AllowPostdatingType_):
    """The http://docs.oasis-open.org/ws-sx/ws-trust/200512/:AllowPostdating element"""
    c_tag = 'AllowPostdating'
    c_namespace = NAMESPACE
    c_children = AllowPostdatingType_.c_children.copy()
    c_attributes = AllowPostdatingType_.c_attributes.copy()
    c_child_order = AllowPostdatingType_.c_child_order[:]
    c_cardinality = AllowPostdatingType_.c_cardinality.copy()