import saml2
from saml2 import SamlBase
from saml2 import saml
class RequestedAttributes(RequestedAttributesType_):
    """The http://eidas.europa.eu/saml-extensions:RequestedAttributes element"""
    c_tag = 'RequestedAttributes'
    c_namespace = NAMESPACE
    c_children = RequestedAttributesType_.c_children.copy()
    c_attributes = RequestedAttributesType_.c_attributes.copy()
    c_child_order = RequestedAttributesType_.c_child_order[:]
    c_cardinality = RequestedAttributesType_.c_cardinality.copy()