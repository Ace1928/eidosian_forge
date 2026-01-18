import saml2
from saml2 import SamlBase
class SPTypeType_(SamlBase):
    """The http://eidas.europa.eu/saml-extensions:SPTypeType element"""
    c_tag = 'SPTypeType'
    c_namespace = NAMESPACE
    c_value_type = {'base': 'xsd:string', 'enumeration': ['public', 'private']}
    c_children = SamlBase.c_children.copy()
    c_attributes = SamlBase.c_attributes.copy()
    c_child_order = SamlBase.c_child_order[:]
    c_cardinality = SamlBase.c_cardinality.copy()