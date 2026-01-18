import saml2
from saml2 import SamlBase
from saml2 import saml
from saml2 import xmldsig as ds
class RequestedAuthnContext(RequestedAuthnContextType_):
    """
    The urn:oasis:names:tc:SAML:2.0:protocol:RequestedAuthnContext element
    """
    c_tag = 'RequestedAuthnContext'
    c_namespace = NAMESPACE
    c_children = RequestedAuthnContextType_.c_children.copy()
    c_attributes = RequestedAuthnContextType_.c_attributes.copy()
    c_child_order = RequestedAuthnContextType_.c_child_order[:]
    c_cardinality = RequestedAuthnContextType_.c_cardinality.copy()