import saml2
from saml2 import SamlBase
class PrincipalAuthenticationMechanism(PrincipalAuthenticationMechanismType_):
    """The urn:oasis:names:tc:SAML:2.0:ac:classes:InternetProtocolPassword:PrincipalAuthenticationMechanism element"""
    c_tag = 'PrincipalAuthenticationMechanism'
    c_namespace = NAMESPACE
    c_children = PrincipalAuthenticationMechanismType_.c_children.copy()
    c_attributes = PrincipalAuthenticationMechanismType_.c_attributes.copy()
    c_child_order = PrincipalAuthenticationMechanismType_.c_child_order[:]
    c_cardinality = PrincipalAuthenticationMechanismType_.c_cardinality.copy()