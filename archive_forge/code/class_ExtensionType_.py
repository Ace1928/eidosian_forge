import saml2
from saml2 import SamlBase
class ExtensionType_(SamlBase):
    """The urn:oasis:names:tc:SAML:2.0:ac:classes:InternetProtocolPassword:ExtensionType element"""
    c_tag = 'ExtensionType'
    c_namespace = NAMESPACE
    c_children = SamlBase.c_children.copy()
    c_attributes = SamlBase.c_attributes.copy()
    c_child_order = SamlBase.c_child_order[:]
    c_cardinality = SamlBase.c_cardinality.copy()