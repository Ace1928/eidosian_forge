import saml2
from saml2 import SamlBase
class ActivationLimitSession(ActivationLimitSessionType_):
    """The urn:oasis:names:tc:SAML:2.0:ac:classes:InternetProtocolPassword:ActivationLimitSession element"""
    c_tag = 'ActivationLimitSession'
    c_namespace = NAMESPACE
    c_children = ActivationLimitSessionType_.c_children.copy()
    c_attributes = ActivationLimitSessionType_.c_attributes.copy()
    c_child_order = ActivationLimitSessionType_.c_child_order[:]
    c_cardinality = ActivationLimitSessionType_.c_cardinality.copy()