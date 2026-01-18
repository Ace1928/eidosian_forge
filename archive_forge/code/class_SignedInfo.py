import saml2
from saml2 import SamlBase
class SignedInfo(SignedInfoType_):
    """The http://www.w3.org/2000/09/xmldsig#:SignedInfo element"""
    c_tag = 'SignedInfo'
    c_namespace = NAMESPACE
    c_children = SignedInfoType_.c_children.copy()
    c_attributes = SignedInfoType_.c_attributes.copy()
    c_child_order = SignedInfoType_.c_child_order[:]
    c_cardinality = SignedInfoType_.c_cardinality.copy()