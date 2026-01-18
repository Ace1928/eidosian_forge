import saml2
from saml2 import SamlBase
class FaultCodesOpenEnumType_(SamlBase):
    """The http://www.w3.org/2005/08/addressing:FaultCodesOpenEnumType element"""
    c_tag = 'FaultCodesOpenEnumType'
    c_namespace = NAMESPACE
    c_children = SamlBase.c_children.copy()
    c_attributes = SamlBase.c_attributes.copy()
    c_child_order = SamlBase.c_child_order[:]
    c_cardinality = SamlBase.c_cardinality.copy()