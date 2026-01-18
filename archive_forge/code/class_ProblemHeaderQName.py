import saml2
from saml2 import SamlBase
class ProblemHeaderQName(AttributedQNameType_):
    """The http://www.w3.org/2005/08/addressing:ProblemHeaderQName element"""
    c_tag = 'ProblemHeaderQName'
    c_namespace = NAMESPACE
    c_children = AttributedQNameType_.c_children.copy()
    c_attributes = AttributedQNameType_.c_attributes.copy()
    c_child_order = AttributedQNameType_.c_child_order[:]
    c_cardinality = AttributedQNameType_.c_cardinality.copy()