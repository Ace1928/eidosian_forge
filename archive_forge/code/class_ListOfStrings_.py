import saml2
from saml2 import SamlBase
from saml2 import md
class ListOfStrings_(SamlBase):
    """The urn:oasis:names:tc:SAML:metadata:ui:listOfStrings element"""
    c_tag = 'listOfStrings'
    c_namespace = NAMESPACE
    c_value_type = {'member': 'string', 'base': 'list'}
    c_children = SamlBase.c_children.copy()
    c_attributes = SamlBase.c_attributes.copy()
    c_child_order = SamlBase.c_child_order[:]
    c_cardinality = SamlBase.c_cardinality.copy()