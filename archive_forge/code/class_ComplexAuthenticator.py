import saml2
from saml2 import SamlBase
class ComplexAuthenticator(ComplexAuthenticatorType_):
    """The urn:oasis:names:tc:SAML:2.0:ac:classes:InternetProtocolPassword:ComplexAuthenticator element"""
    c_tag = 'ComplexAuthenticator'
    c_namespace = NAMESPACE
    c_children = ComplexAuthenticatorType_.c_children.copy()
    c_attributes = ComplexAuthenticatorType_.c_attributes.copy()
    c_child_order = ComplexAuthenticatorType_.c_child_order[:]
    c_cardinality = ComplexAuthenticatorType_.c_cardinality.copy()