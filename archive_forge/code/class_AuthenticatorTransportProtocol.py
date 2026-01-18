import saml2
from saml2 import SamlBase
class AuthenticatorTransportProtocol(AuthenticatorTransportProtocolType_):
    """The urn:oasis:names:tc:SAML:2.0:ac:classes:InternetProtocolPassword:AuthenticatorTransportProtocol element"""
    c_tag = 'AuthenticatorTransportProtocol'
    c_namespace = NAMESPACE
    c_children = AuthenticatorTransportProtocolType_.c_children.copy()
    c_attributes = AuthenticatorTransportProtocolType_.c_attributes.copy()
    c_child_order = AuthenticatorTransportProtocolType_.c_child_order[:]
    c_cardinality = AuthenticatorTransportProtocolType_.c_cardinality.copy()