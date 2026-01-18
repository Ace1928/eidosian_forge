import saml2
from saml2 import SamlBase
class ActivationLimitDuration(ActivationLimitDurationType_):
    """The urn:oasis:names:tc:SAML:2.0:ac:classes:InternetProtocolPassword:ActivationLimitDuration element"""
    c_tag = 'ActivationLimitDuration'
    c_namespace = NAMESPACE
    c_children = ActivationLimitDurationType_.c_children.copy()
    c_attributes = ActivationLimitDurationType_.c_attributes.copy()
    c_child_order = ActivationLimitDurationType_.c_child_order[:]
    c_cardinality = ActivationLimitDurationType_.c_cardinality.copy()