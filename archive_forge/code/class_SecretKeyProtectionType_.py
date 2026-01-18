import saml2
from saml2 import SamlBase
class SecretKeyProtectionType_(SamlBase):
    """The urn:oasis:names:tc:SAML:2.0:ac:classes:InternetProtocolPassword:SecretKeyProtectionType element"""
    c_tag = 'SecretKeyProtectionType'
    c_namespace = NAMESPACE
    c_children = SamlBase.c_children.copy()
    c_attributes = SamlBase.c_attributes.copy()
    c_child_order = SamlBase.c_child_order[:]
    c_cardinality = SamlBase.c_cardinality.copy()
    c_children['{urn:oasis:names:tc:SAML:2.0:ac:classes:InternetProtocolPassword}KeyActivation'] = ('key_activation', KeyActivation)
    c_cardinality['key_activation'] = {'min': 0, 'max': 1}
    c_children['{urn:oasis:names:tc:SAML:2.0:ac:classes:InternetProtocolPassword}KeyStorage'] = ('key_storage', KeyStorage)
    c_cardinality['key_storage'] = {'min': 0, 'max': 1}
    c_children['{urn:oasis:names:tc:SAML:2.0:ac:classes:InternetProtocolPassword}Extension'] = ('extension', [Extension])
    c_cardinality['extension'] = {'min': 0}
    c_child_order.extend(['key_activation', 'key_storage', 'extension'])

    def __init__(self, key_activation=None, key_storage=None, extension=None, text=None, extension_elements=None, extension_attributes=None):
        SamlBase.__init__(self, text=text, extension_elements=extension_elements, extension_attributes=extension_attributes)
        self.key_activation = key_activation
        self.key_storage = key_storage
        self.extension = extension or []