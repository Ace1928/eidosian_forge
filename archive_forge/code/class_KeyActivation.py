import saml2
from saml2 import SamlBase
class KeyActivation(KeyActivationType_):
    """The urn:oasis:names:tc:SAML:2.0:ac:classes:InternetProtocolPassword:KeyActivation element"""
    c_tag = 'KeyActivation'
    c_namespace = NAMESPACE
    c_children = KeyActivationType_.c_children.copy()
    c_attributes = KeyActivationType_.c_attributes.copy()
    c_child_order = KeyActivationType_.c_child_order[:]
    c_cardinality = KeyActivationType_.c_cardinality.copy()