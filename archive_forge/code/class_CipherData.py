import saml2
from saml2 import SamlBase
from saml2 import xmldsig as ds
class CipherData(CipherDataType_):
    """The http://www.w3.org/2001/04/xmlenc#:CipherData element"""
    c_tag = 'CipherData'
    c_namespace = NAMESPACE
    c_children = CipherDataType_.c_children.copy()
    c_attributes = CipherDataType_.c_attributes.copy()
    c_child_order = CipherDataType_.c_child_order[:]
    c_cardinality = CipherDataType_.c_cardinality.copy()