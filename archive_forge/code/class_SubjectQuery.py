import saml2
from saml2 import SamlBase
from saml2 import saml
from saml2 import xmldsig as ds
class SubjectQuery(SubjectQueryAbstractType_):
    """The urn:oasis:names:tc:SAML:2.0:protocol:SubjectQuery element"""
    c_tag = 'SubjectQuery'
    c_namespace = NAMESPACE
    c_children = SubjectQueryAbstractType_.c_children.copy()
    c_attributes = SubjectQueryAbstractType_.c_attributes.copy()
    c_child_order = SubjectQueryAbstractType_.c_child_order[:]
    c_cardinality = SubjectQueryAbstractType_.c_cardinality.copy()