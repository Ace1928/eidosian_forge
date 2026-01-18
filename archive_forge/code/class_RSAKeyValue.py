import saml2
from saml2 import SamlBase
class RSAKeyValue(RSAKeyValueType_):
    """The http://www.w3.org/2000/09/xmldsig#:RSAKeyValue element"""
    c_tag = 'RSAKeyValue'
    c_namespace = NAMESPACE
    c_children = RSAKeyValueType_.c_children.copy()
    c_attributes = RSAKeyValueType_.c_attributes.copy()
    c_child_order = RSAKeyValueType_.c_child_order[:]
    c_cardinality = RSAKeyValueType_.c_cardinality.copy()