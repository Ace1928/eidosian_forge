import saml2
from saml2 import SamlBase
class KeyActivationType_(SamlBase):
    """The urn:oasis:names:tc:SAML:2.0:ac:classes:InternetProtocolPassword:KeyActivationType element"""
    c_tag = 'KeyActivationType'
    c_namespace = NAMESPACE
    c_children = SamlBase.c_children.copy()
    c_attributes = SamlBase.c_attributes.copy()
    c_child_order = SamlBase.c_child_order[:]
    c_cardinality = SamlBase.c_cardinality.copy()
    c_children['{urn:oasis:names:tc:SAML:2.0:ac:classes:InternetProtocolPassword}ActivationPin'] = ('activation_pin', ActivationPin)
    c_cardinality['activation_pin'] = {'min': 0, 'max': 1}
    c_children['{urn:oasis:names:tc:SAML:2.0:ac:classes:InternetProtocolPassword}Extension'] = ('extension', [Extension])
    c_cardinality['extension'] = {'min': 0}
    c_child_order.extend(['activation_pin', 'extension'])

    def __init__(self, activation_pin=None, extension=None, text=None, extension_elements=None, extension_attributes=None):
        SamlBase.__init__(self, text=text, extension_elements=extension_elements, extension_attributes=extension_attributes)
        self.activation_pin = activation_pin
        self.extension = extension or []