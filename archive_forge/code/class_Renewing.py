import saml2
from saml2 import SamlBase
from saml2.ws import wsaddr as wsa
from saml2.ws import wssec as wsse
from saml2.ws import wsutil as wsu
class Renewing(RenewingType_):
    """The http://docs.oasis-open.org/ws-sx/ws-trust/200512/:Renewing element"""
    c_tag = 'Renewing'
    c_namespace = NAMESPACE
    c_children = RenewingType_.c_children.copy()
    c_attributes = RenewingType_.c_attributes.copy()
    c_child_order = RenewingType_.c_child_order[:]
    c_cardinality = RenewingType_.c_cardinality.copy()