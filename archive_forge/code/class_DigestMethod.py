import saml2
from saml2 import SamlBase
class DigestMethod(DigestMethodType_):
    """The urn:oasis:names:tc:SAML:metadata:algsupport:DigestMethod element"""
    c_tag = 'DigestMethod'
    c_namespace = NAMESPACE
    c_children = DigestMethodType_.c_children.copy()
    c_attributes = DigestMethodType_.c_attributes.copy()
    c_child_order = DigestMethodType_.c_child_order[:]
    c_cardinality = DigestMethodType_.c_cardinality.copy()