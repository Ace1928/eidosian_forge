import saml2
from saml2 import SamlBase
class RetryAfter(AttributedUnsignedLongType_):
    """The http://www.w3.org/2005/08/addressing:RetryAfter element"""
    c_tag = 'RetryAfter'
    c_namespace = NAMESPACE
    c_children = AttributedUnsignedLongType_.c_children.copy()
    c_attributes = AttributedUnsignedLongType_.c_attributes.copy()
    c_child_order = AttributedUnsignedLongType_.c_child_order[:]
    c_cardinality = AttributedUnsignedLongType_.c_cardinality.copy()