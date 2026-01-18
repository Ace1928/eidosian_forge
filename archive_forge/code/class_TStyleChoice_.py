import saml2
from saml2 import SamlBase
from saml2.schema import wsdl
class TStyleChoice_(SamlBase):
    """The http://schemas.xmlsoap.org/wsdl/soap/:tStyleChoice element"""
    c_tag = 'tStyleChoice'
    c_namespace = NAMESPACE
    c_value_type = {'base': 'xs:string', 'enumeration': ['rpc', 'document']}
    c_children = SamlBase.c_children.copy()
    c_attributes = SamlBase.c_attributes.copy()
    c_child_order = SamlBase.c_child_order[:]
    c_cardinality = SamlBase.c_cardinality.copy()