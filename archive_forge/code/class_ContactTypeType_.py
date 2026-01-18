import saml2
from saml2 import SamlBase
from saml2 import saml
from saml2 import xmldsig as ds
from saml2 import xmlenc as xenc
class ContactTypeType_(SamlBase):
    """The urn:oasis:names:tc:SAML:2.0:metadata:ContactTypeType element"""
    c_tag = 'ContactTypeType'
    c_namespace = NAMESPACE
    c_value_type = {'base': 'string', 'enumeration': ['technical', 'support', 'administrative', 'billing', 'other']}
    c_children = SamlBase.c_children.copy()
    c_attributes = SamlBase.c_attributes.copy()
    c_child_order = SamlBase.c_child_order[:]
    c_cardinality = SamlBase.c_cardinality.copy()