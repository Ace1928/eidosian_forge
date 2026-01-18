import saml2
from saml2 import SamlBase
from saml2 import saml
from saml2 import xmldsig as ds
from saml2 import xmlenc as xenc
class AttributeConsumingService(AttributeConsumingServiceType_):
    """The urn:oasis:names:tc:SAML:2.0:metadata:AttributeConsumingService
    element"""
    c_tag = 'AttributeConsumingService'
    c_namespace = NAMESPACE
    c_children = AttributeConsumingServiceType_.c_children.copy()
    c_attributes = AttributeConsumingServiceType_.c_attributes.copy()
    c_child_order = AttributeConsumingServiceType_.c_child_order[:]
    c_cardinality = AttributeConsumingServiceType_.c_cardinality.copy()