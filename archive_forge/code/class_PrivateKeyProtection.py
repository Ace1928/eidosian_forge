import saml2
from saml2 import SamlBase
class PrivateKeyProtection(PrivateKeyProtectionType_):
    """The urn:oasis:names:tc:SAML:2.0:ac:classes:InternetProtocolPassword:PrivateKeyProtection element"""
    c_tag = 'PrivateKeyProtection'
    c_namespace = NAMESPACE
    c_children = PrivateKeyProtectionType_.c_children.copy()
    c_attributes = PrivateKeyProtectionType_.c_attributes.copy()
    c_child_order = PrivateKeyProtectionType_.c_child_order[:]
    c_cardinality = PrivateKeyProtectionType_.c_cardinality.copy()