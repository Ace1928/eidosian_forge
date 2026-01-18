import saml2
from saml2 import SamlBase
from saml2 import xmldsig as ds
class EncryptionProperties(EncryptionPropertiesType_):
    """The http://www.w3.org/2001/04/xmlenc#:EncryptionProperties element"""
    c_tag = 'EncryptionProperties'
    c_namespace = NAMESPACE
    c_children = EncryptionPropertiesType_.c_children.copy()
    c_attributes = EncryptionPropertiesType_.c_attributes.copy()
    c_child_order = EncryptionPropertiesType_.c_child_order[:]
    c_cardinality = EncryptionPropertiesType_.c_cardinality.copy()