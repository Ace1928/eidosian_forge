import saml2
from saml2 import SamlBase
class X509Data(X509DataType_):
    """The http://www.w3.org/2000/09/xmldsig#:X509Data element"""
    c_tag = 'X509Data'
    c_namespace = NAMESPACE
    c_children = X509DataType_.c_children.copy()
    c_attributes = X509DataType_.c_attributes.copy()
    c_child_order = X509DataType_.c_child_order[:]
    c_cardinality = X509DataType_.c_cardinality.copy()