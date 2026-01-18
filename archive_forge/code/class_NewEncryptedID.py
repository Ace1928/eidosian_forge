import saml2
from saml2 import SamlBase
from saml2 import saml
from saml2 import xmldsig as ds
class NewEncryptedID(saml.EncryptedElementType_):
    """The urn:oasis:names:tc:SAML:2.0:protocol:NewEncryptedID element"""
    c_tag = 'NewEncryptedID'
    c_namespace = NAMESPACE
    c_children = saml.EncryptedElementType_.c_children.copy()
    c_attributes = saml.EncryptedElementType_.c_attributes.copy()
    c_child_order = saml.EncryptedElementType_.c_child_order[:]
    c_cardinality = saml.EncryptedElementType_.c_cardinality.copy()