import saml2
from saml2 import SamlBase
class KeyStorage(KeyStorageType_):
    """The urn:oasis:names:tc:SAML:2.0:ac:classes:InternetProtocolPassword:KeyStorage element"""
    c_tag = 'KeyStorage'
    c_namespace = NAMESPACE
    c_children = KeyStorageType_.c_children.copy()
    c_attributes = KeyStorageType_.c_attributes.copy()
    c_child_order = KeyStorageType_.c_child_order[:]
    c_cardinality = KeyStorageType_.c_cardinality.copy()