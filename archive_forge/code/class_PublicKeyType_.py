import saml2
from saml2 import SamlBase
class PublicKeyType_(SamlBase):
    """The urn:oasis:names:tc:SAML:2.0:ac:classes:InternetProtocolPassword:PublicKeyType element"""
    c_tag = 'PublicKeyType'
    c_namespace = NAMESPACE
    c_children = SamlBase.c_children.copy()
    c_attributes = SamlBase.c_attributes.copy()
    c_child_order = SamlBase.c_child_order[:]
    c_cardinality = SamlBase.c_cardinality.copy()
    c_children['{urn:oasis:names:tc:SAML:2.0:ac:classes:InternetProtocolPassword}Extension'] = ('extension', [Extension])
    c_cardinality['extension'] = {'min': 0}
    c_attributes['keyValidation'] = ('key_validation', 'None', False)
    c_child_order.extend(['extension'])

    def __init__(self, extension=None, key_validation=None, text=None, extension_elements=None, extension_attributes=None):
        SamlBase.__init__(self, text=text, extension_elements=extension_elements, extension_attributes=extension_attributes)
        self.extension = extension or []
        self.key_validation = key_validation