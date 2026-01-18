import saml2
from saml2 import SamlBase
class DeviceTypeType_(SamlBase):
    """The urn:oasis:names:tc:SAML:2.0:ac:classes:InternetProtocolPassword:DeviceTypeType element"""
    c_tag = 'DeviceTypeType'
    c_namespace = NAMESPACE
    c_value_type = {'base': 'xs:NMTOKEN', 'enumeration': ['hardware', 'software']}
    c_children = SamlBase.c_children.copy()
    c_attributes = SamlBase.c_attributes.copy()
    c_child_order = SamlBase.c_child_order[:]
    c_cardinality = SamlBase.c_cardinality.copy()