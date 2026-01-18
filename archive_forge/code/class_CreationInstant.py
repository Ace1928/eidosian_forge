import saml2
from saml2 import SamlBase
from saml2 import md
class CreationInstant(SamlBase):
    """The urn:oasis:names:tc:SAML:2.0:metadata:dri:CreationInstant element"""
    c_tag = 'CreationInstant'
    c_namespace = NAMESPACE
    c_value_type = {'base': 'datetime'}
    c_children = SamlBase.c_children.copy()
    c_attributes = SamlBase.c_attributes.copy()
    c_child_order = SamlBase.c_child_order[:]
    c_cardinality = SamlBase.c_cardinality.copy()