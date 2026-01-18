import saml2
from saml2 import SamlBase
from saml2 import saml
from saml2 import xmldsig as ds
class LogoutRequest(LogoutRequestType_):
    """The urn:oasis:names:tc:SAML:2.0:protocol:LogoutRequest element"""
    c_tag = 'LogoutRequest'
    c_namespace = NAMESPACE
    c_children = LogoutRequestType_.c_children.copy()
    c_attributes = LogoutRequestType_.c_attributes.copy()
    c_child_order = LogoutRequestType_.c_child_order[:]
    c_cardinality = LogoutRequestType_.c_cardinality.copy()