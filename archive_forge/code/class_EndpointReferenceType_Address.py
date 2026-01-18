import saml2
from saml2 import SamlBase
class EndpointReferenceType_Address(AttributedURIType_):
    c_tag = 'Address'
    c_namespace = NAMESPACE
    c_children = AttributedURIType_.c_children.copy()
    c_attributes = AttributedURIType_.c_attributes.copy()
    c_child_order = AttributedURIType_.c_child_order[:]
    c_cardinality = AttributedURIType_.c_cardinality.copy()