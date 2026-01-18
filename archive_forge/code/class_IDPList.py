import saml2
from saml2 import SamlBase
from saml2 import saml
from saml2 import xmldsig as ds
class IDPList(IDPListType_):
    """The urn:oasis:names:tc:SAML:2.0:protocol:IDPList element"""
    c_tag = 'IDPList'
    c_namespace = NAMESPACE
    c_children = IDPListType_.c_children.copy()
    c_attributes = IDPListType_.c_attributes.copy()
    c_child_order = IDPListType_.c_child_order[:]
    c_cardinality = IDPListType_.c_cardinality.copy()