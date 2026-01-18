import saml2
from saml2 import SamlBase
from saml2 import xmldsig as ds
class CipherReference(CipherReferenceType_):
    """The http://www.w3.org/2001/04/xmlenc#:CipherReference element"""
    c_tag = 'CipherReference'
    c_namespace = NAMESPACE
    c_children = CipherReferenceType_.c_children.copy()
    c_attributes = CipherReferenceType_.c_attributes.copy()
    c_child_order = CipherReferenceType_.c_child_order[:]
    c_cardinality = CipherReferenceType_.c_cardinality.copy()