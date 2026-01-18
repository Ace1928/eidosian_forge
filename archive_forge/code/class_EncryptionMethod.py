import saml2
from saml2 import SamlBase
from saml2 import xmldsig as ds
class EncryptionMethod(EncryptionMethodType_):
    c_tag = 'EncryptionMethod'
    c_namespace = NAMESPACE
    c_children = EncryptionMethodType_.c_children.copy()
    c_attributes = EncryptionMethodType_.c_attributes.copy()
    c_child_order = EncryptionMethodType_.c_child_order[:]
    c_cardinality = EncryptionMethodType_.c_cardinality.copy()