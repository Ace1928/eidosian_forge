import saml2
from saml2 import SamlBase
class AppliesTo(SamlBase):
    """The http://schemas.xmlsoap.org/ws/2004/09/policy:AppliesTo element"""
    c_tag = 'AppliesTo'
    c_namespace = NAMESPACE
    c_children = SamlBase.c_children.copy()
    c_attributes = SamlBase.c_attributes.copy()
    c_child_order = SamlBase.c_child_order[:]
    c_cardinality = SamlBase.c_cardinality.copy()