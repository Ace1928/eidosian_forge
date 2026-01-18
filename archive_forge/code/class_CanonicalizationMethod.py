import saml2
from saml2 import SamlBase
class CanonicalizationMethod(CanonicalizationMethodType_):
    """The http://www.w3.org/2000/09/xmldsig#:CanonicalizationMethod element"""
    c_tag = 'CanonicalizationMethod'
    c_namespace = NAMESPACE
    c_children = CanonicalizationMethodType_.c_children.copy()
    c_attributes = CanonicalizationMethodType_.c_attributes.copy()
    c_child_order = CanonicalizationMethodType_.c_child_order[:]
    c_cardinality = CanonicalizationMethodType_.c_cardinality.copy()