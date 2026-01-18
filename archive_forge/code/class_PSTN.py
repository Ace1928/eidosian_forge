import saml2
from saml2 import SamlBase
class PSTN(ExtensionOnlyType_):
    """The urn:oasis:names:tc:SAML:2.0:ac:classes:InternetProtocolPassword:PSTN element"""
    c_tag = 'PSTN'
    c_namespace = NAMESPACE
    c_children = ExtensionOnlyType_.c_children.copy()
    c_attributes = ExtensionOnlyType_.c_attributes.copy()
    c_child_order = ExtensionOnlyType_.c_child_order[:]
    c_cardinality = ExtensionOnlyType_.c_cardinality.copy()