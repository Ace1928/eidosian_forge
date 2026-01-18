import saml2
from saml2 import SamlBase
from saml2.schema import wsdl
class UseChoice_(SamlBase):
    """The http://schemas.xmlsoap.org/wsdl/soap/:useChoice element"""
    c_tag = 'useChoice'
    c_namespace = NAMESPACE
    c_value_type = {'base': 'xs:string', 'enumeration': ['literal', 'encoded']}
    c_children = SamlBase.c_children.copy()
    c_attributes = SamlBase.c_attributes.copy()
    c_child_order = SamlBase.c_child_order[:]
    c_cardinality = SamlBase.c_cardinality.copy()