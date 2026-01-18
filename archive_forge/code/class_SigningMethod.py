import saml2
from saml2 import SamlBase
class SigningMethod(SigningMethodType_):
    """The urn:oasis:names:tc:SAML:metadata:algsupport:SigningMethod element"""
    c_tag = 'SigningMethod'
    c_namespace = NAMESPACE
    c_children = SigningMethodType_.c_children.copy()
    c_attributes = SigningMethodType_.c_attributes.copy()
    c_child_order = SigningMethodType_.c_child_order[:]
    c_cardinality = SigningMethodType_.c_cardinality.copy()