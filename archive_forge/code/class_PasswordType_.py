import saml2
from saml2 import SamlBase
class PasswordType_(SamlBase):
    """The urn:oasis:names:tc:SAML:2.0:ac:classes:InternetProtocolPassword:PasswordType element"""
    c_tag = 'PasswordType'
    c_namespace = NAMESPACE
    c_children = SamlBase.c_children.copy()
    c_attributes = SamlBase.c_attributes.copy()
    c_child_order = SamlBase.c_child_order[:]
    c_cardinality = SamlBase.c_cardinality.copy()
    c_children['{urn:oasis:names:tc:SAML:2.0:ac:classes:InternetProtocolPassword}Length'] = ('length', Length)
    c_cardinality['length'] = {'min': 0, 'max': 1}
    c_children['{urn:oasis:names:tc:SAML:2.0:ac:classes:InternetProtocolPassword}Alphabet'] = ('alphabet', Alphabet)
    c_cardinality['alphabet'] = {'min': 0, 'max': 1}
    c_children['{urn:oasis:names:tc:SAML:2.0:ac:classes:InternetProtocolPassword}Generation'] = ('generation', Generation)
    c_cardinality['generation'] = {'min': 0, 'max': 1}
    c_children['{urn:oasis:names:tc:SAML:2.0:ac:classes:InternetProtocolPassword}Extension'] = ('extension', [Extension])
    c_cardinality['extension'] = {'min': 0}
    c_attributes['ExternalVerification'] = ('external_verification', 'anyURI', False)
    c_child_order.extend(['length', 'alphabet', 'generation', 'extension'])

    def __init__(self, length=None, alphabet=None, generation=None, extension=None, external_verification=None, text=None, extension_elements=None, extension_attributes=None):
        SamlBase.__init__(self, text=text, extension_elements=extension_elements, extension_attributes=extension_attributes)
        self.length = length
        self.alphabet = alphabet
        self.generation = generation
        self.extension = extension or []
        self.external_verification = external_verification