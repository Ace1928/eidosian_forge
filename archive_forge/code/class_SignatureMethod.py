import saml2
from saml2 import SamlBase
class SignatureMethod(SignatureMethodType_):
    """The http://www.w3.org/2000/09/xmldsig#:SignatureMethod element"""
    c_tag = 'SignatureMethod'
    c_namespace = NAMESPACE
    c_children = SignatureMethodType_.c_children.copy()
    c_attributes = SignatureMethodType_.c_attributes.copy()
    c_child_order = SignatureMethodType_.c_child_order[:]
    c_cardinality = SignatureMethodType_.c_cardinality.copy()