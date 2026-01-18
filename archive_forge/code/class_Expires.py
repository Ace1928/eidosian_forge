import saml2
from saml2 import SamlBase
class Expires(AttributedDateTime_):
    """The http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-wssecurity-utility-1.0.xsd:Expires element"""
    c_tag = 'Expires'
    c_namespace = NAMESPACE
    c_children = AttributedDateTime_.c_children.copy()
    c_attributes = AttributedDateTime_.c_attributes.copy()
    c_child_order = AttributedDateTime_.c_child_order[:]
    c_cardinality = AttributedDateTime_.c_cardinality.copy()