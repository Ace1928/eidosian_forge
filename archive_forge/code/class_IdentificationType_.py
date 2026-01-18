import saml2
from saml2 import SamlBase
class IdentificationType_(SamlBase):
    """The urn:oasis:names:tc:SAML:2.0:ac:classes:InternetProtocolPassword:IdentificationType element"""
    c_tag = 'IdentificationType'
    c_namespace = NAMESPACE
    c_children = SamlBase.c_children.copy()
    c_attributes = SamlBase.c_attributes.copy()
    c_child_order = SamlBase.c_child_order[:]
    c_cardinality = SamlBase.c_cardinality.copy()
    c_children['{urn:oasis:names:tc:SAML:2.0:ac:classes:InternetProtocolPassword}PhysicalVerification'] = ('physical_verification', PhysicalVerification)
    c_cardinality['physical_verification'] = {'min': 0, 'max': 1}
    c_children['{urn:oasis:names:tc:SAML:2.0:ac:classes:InternetProtocolPassword}WrittenConsent'] = ('written_consent', WrittenConsent)
    c_cardinality['written_consent'] = {'min': 0, 'max': 1}
    c_children['{urn:oasis:names:tc:SAML:2.0:ac:classes:InternetProtocolPassword}GoverningAgreements'] = ('governing_agreements', GoverningAgreements)
    c_cardinality['governing_agreements'] = {'min': 0, 'max': 1}
    c_children['{urn:oasis:names:tc:SAML:2.0:ac:classes:InternetProtocolPassword}Extension'] = ('extension', [Extension])
    c_cardinality['extension'] = {'min': 0}
    c_attributes['nym'] = ('nym', NymType_, False)
    c_child_order.extend(['physical_verification', 'written_consent', 'governing_agreements', 'extension'])

    def __init__(self, physical_verification=None, written_consent=None, governing_agreements=None, extension=None, nym=None, text=None, extension_elements=None, extension_attributes=None):
        SamlBase.__init__(self, text=text, extension_elements=extension_elements, extension_attributes=extension_attributes)
        self.physical_verification = physical_verification
        self.written_consent = written_consent
        self.governing_agreements = governing_agreements
        self.extension = extension or []
        self.nym = nym