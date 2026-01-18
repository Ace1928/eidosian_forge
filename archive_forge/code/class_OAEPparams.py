import saml2
from saml2 import SamlBase
from saml2 import xmldsig as ds
class OAEPparams(SamlBase):
    c_tag = 'OAEPparams'
    c_namespace = NAMESPACE
    c_value_type = {'base': 'base64Binary'}
    c_children = SamlBase.c_children.copy()
    c_attributes = SamlBase.c_attributes.copy()
    c_child_order = SamlBase.c_child_order[:]
    c_cardinality = SamlBase.c_cardinality.copy()