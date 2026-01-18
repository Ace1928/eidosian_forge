import saml2
from saml2 import SamlBase
class RelatesTo(RelatesToType_):
    """The http://www.w3.org/2005/08/addressing:RelatesTo element"""
    c_tag = 'RelatesTo'
    c_namespace = NAMESPACE
    c_children = RelatesToType_.c_children.copy()
    c_attributes = RelatesToType_.c_attributes.copy()
    c_child_order = RelatesToType_.c_child_order[:]
    c_cardinality = RelatesToType_.c_cardinality.copy()