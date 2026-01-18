import saml2
from saml2 import SamlBase
class AuthnMethod(AuthnMethodBaseType_):
    """The urn:oasis:names:tc:SAML:2.0:ac:classes:InternetProtocolPassword:AuthnMethod element"""
    c_tag = 'AuthnMethod'
    c_namespace = NAMESPACE
    c_children = AuthnMethodBaseType_.c_children.copy()
    c_attributes = AuthnMethodBaseType_.c_attributes.copy()
    c_child_order = AuthnMethodBaseType_.c_child_order[:]
    c_cardinality = AuthnMethodBaseType_.c_cardinality.copy()