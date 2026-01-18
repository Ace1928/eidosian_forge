import saml2
from saml2 import SamlBase
class ActivationLimitType_(SamlBase):
    """The urn:oasis:names:tc:SAML:2.0:ac:classes:InternetProtocolPassword:ActivationLimitType element"""
    c_tag = 'ActivationLimitType'
    c_namespace = NAMESPACE
    c_children = SamlBase.c_children.copy()
    c_attributes = SamlBase.c_attributes.copy()
    c_child_order = SamlBase.c_child_order[:]
    c_cardinality = SamlBase.c_cardinality.copy()
    c_children['{urn:oasis:names:tc:SAML:2.0:ac:classes:InternetProtocolPassword}ActivationLimitDuration'] = ('activation_limit_duration', ActivationLimitDuration)
    c_cardinality['activation_limit_duration'] = {'min': 0, 'max': 1}
    c_children['{urn:oasis:names:tc:SAML:2.0:ac:classes:InternetProtocolPassword}ActivationLimitUsages'] = ('activation_limit_usages', ActivationLimitUsages)
    c_cardinality['activation_limit_usages'] = {'min': 0, 'max': 1}
    c_children['{urn:oasis:names:tc:SAML:2.0:ac:classes:InternetProtocolPassword}ActivationLimitSession'] = ('activation_limit_session', ActivationLimitSession)
    c_cardinality['activation_limit_session'] = {'min': 0, 'max': 1}
    c_child_order.extend(['activation_limit_duration', 'activation_limit_usages', 'activation_limit_session'])

    def __init__(self, activation_limit_duration=None, activation_limit_usages=None, activation_limit_session=None, text=None, extension_elements=None, extension_attributes=None):
        SamlBase.__init__(self, text=text, extension_elements=extension_elements, extension_attributes=extension_attributes)
        self.activation_limit_duration = activation_limit_duration
        self.activation_limit_usages = activation_limit_usages
        self.activation_limit_session = activation_limit_session