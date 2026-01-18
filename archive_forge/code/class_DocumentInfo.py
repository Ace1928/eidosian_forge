import saml2
from saml2 import SamlBase
from saml2 import md
class DocumentInfo(DocumentInfoType_):
    """The urn:oasis:names:tc:SAML:2.0:metadata:dri:DocumentInfo element"""
    c_tag = 'DocumentInfo'
    c_namespace = NAMESPACE
    c_children = DocumentInfoType_.c_children.copy()
    c_attributes = DocumentInfoType_.c_attributes.copy()
    c_child_order = DocumentInfoType_.c_child_order[:]
    c_cardinality = DocumentInfoType_.c_cardinality.copy()