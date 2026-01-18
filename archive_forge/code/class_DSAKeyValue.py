import saml2
from saml2 import SamlBase
class DSAKeyValue(DSAKeyValueType_):
    """The http://www.w3.org/2000/09/xmldsig#:DSAKeyValue element"""
    c_tag = 'DSAKeyValue'
    c_namespace = NAMESPACE
    c_children = DSAKeyValueType_.c_children.copy()
    c_attributes = DSAKeyValueType_.c_attributes.copy()
    c_child_order = DSAKeyValueType_.c_child_order[:]
    c_cardinality = DSAKeyValueType_.c_cardinality.copy()