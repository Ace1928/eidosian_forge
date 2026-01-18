import saml2
from saml2 import SamlBase
class SPType(SPTypeType_):
    """The http://eidas.europa.eu/saml-extensions:SPType element"""
    c_tag = 'SPType'
    c_namespace = NAMESPACE
    c_children = SPTypeType_.c_children.copy()
    c_attributes = SPTypeType_.c_attributes.copy()
    c_child_order = SPTypeType_.c_child_order[:]
    c_cardinality = SPTypeType_.c_cardinality.copy()