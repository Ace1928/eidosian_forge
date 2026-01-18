import saml2
from saml2 import SamlBase
from saml2 import xmldsig as ds
class EncryptedDataType_(EncryptedType_):
    """The http://www.w3.org/2001/04/xmlenc#:EncryptedDataType element"""
    c_tag = 'EncryptedDataType'
    c_namespace = NAMESPACE
    c_children = EncryptedType_.c_children.copy()
    c_attributes = EncryptedType_.c_attributes.copy()
    c_child_order = EncryptedType_.c_child_order[:]
    c_cardinality = EncryptedType_.c_cardinality.copy()