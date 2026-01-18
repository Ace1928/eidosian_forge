import saml2
from saml2 import SamlBase
class SignatureProperty(SignaturePropertyType_):
    """The http://www.w3.org/2000/09/xmldsig#:SignatureProperty element"""
    c_tag = 'SignatureProperty'
    c_namespace = NAMESPACE
    c_children = SignaturePropertyType_.c_children.copy()
    c_attributes = SignaturePropertyType_.c_attributes.copy()
    c_child_order = SignaturePropertyType_.c_child_order[:]
    c_cardinality = SignaturePropertyType_.c_cardinality.copy()