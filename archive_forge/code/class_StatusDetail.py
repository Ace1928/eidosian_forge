import saml2
from saml2 import SamlBase
from saml2 import saml
from saml2 import xmldsig as ds
class StatusDetail(StatusDetailType_):
    """The urn:oasis:names:tc:SAML:2.0:protocol:StatusDetail element"""
    c_tag = 'StatusDetail'
    c_namespace = NAMESPACE
    c_children = StatusDetailType_.c_children.copy()
    c_attributes = StatusDetailType_.c_attributes.copy()
    c_child_order = StatusDetailType_.c_child_order[:]
    c_cardinality = StatusDetailType_.c_cardinality.copy()