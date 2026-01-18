import saml2
from saml2 import SamlBase
class TechnicalProtection(TechnicalProtectionBaseType_):
    """The urn:oasis:names:tc:SAML:2.0:ac:classes:InternetProtocolPassword:TechnicalProtection element"""
    c_tag = 'TechnicalProtection'
    c_namespace = NAMESPACE
    c_children = TechnicalProtectionBaseType_.c_children.copy()
    c_attributes = TechnicalProtectionBaseType_.c_attributes.copy()
    c_child_order = TechnicalProtectionBaseType_.c_child_order[:]
    c_cardinality = TechnicalProtectionBaseType_.c_cardinality.copy()