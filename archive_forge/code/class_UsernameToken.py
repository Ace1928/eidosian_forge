import saml2
from saml2 import SamlBase
class UsernameToken(UsernameTokenType_):
    """The http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-wssecurity-secext-1.0.xsd:UsernameToken element"""
    c_tag = 'UsernameToken'
    c_namespace = NAMESPACE
    c_children = UsernameTokenType_.c_children.copy()
    c_attributes = UsernameTokenType_.c_attributes.copy()
    c_child_order = UsernameTokenType_.c_child_order[:]
    c_cardinality = UsernameTokenType_.c_cardinality.copy()