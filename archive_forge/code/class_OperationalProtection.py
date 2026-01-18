import saml2
from saml2 import SamlBase
class OperationalProtection(OperationalProtectionType_):
    """The urn:oasis:names:tc:SAML:2.0:ac:classes:InternetProtocolPassword:OperationalProtection element"""
    c_tag = 'OperationalProtection'
    c_namespace = NAMESPACE
    c_children = OperationalProtectionType_.c_children.copy()
    c_attributes = OperationalProtectionType_.c_attributes.copy()
    c_child_order = OperationalProtectionType_.c_child_order[:]
    c_cardinality = OperationalProtectionType_.c_cardinality.copy()