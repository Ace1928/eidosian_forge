import saml2
from saml2 import SamlBase
class ReplyTo(EndpointReferenceType_):
    """The http://www.w3.org/2005/08/addressing:ReplyTo element"""
    c_tag = 'ReplyTo'
    c_namespace = NAMESPACE
    c_children = EndpointReferenceType_.c_children.copy()
    c_attributes = EndpointReferenceType_.c_attributes.copy()
    c_child_order = EndpointReferenceType_.c_child_order[:]
    c_cardinality = EndpointReferenceType_.c_cardinality.copy()