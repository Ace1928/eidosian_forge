import saml2
from saml2 import SamlBase
class RetrievalMethod(RetrievalMethodType_):
    """The http://www.w3.org/2000/09/xmldsig#:RetrievalMethod element"""
    c_tag = 'RetrievalMethod'
    c_namespace = NAMESPACE
    c_children = RetrievalMethodType_.c_children.copy()
    c_attributes = RetrievalMethodType_.c_attributes.copy()
    c_child_order = RetrievalMethodType_.c_child_order[:]
    c_cardinality = RetrievalMethodType_.c_cardinality.copy()