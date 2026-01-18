import saml2
from saml2 import SamlBase
class GoverningAgreementsType_(SamlBase):
    """The urn:oasis:names:tc:SAML:2.0:ac:classes:InternetProtocolPassword:GoverningAgreementsType element"""
    c_tag = 'GoverningAgreementsType'
    c_namespace = NAMESPACE
    c_children = SamlBase.c_children.copy()
    c_attributes = SamlBase.c_attributes.copy()
    c_child_order = SamlBase.c_child_order[:]
    c_cardinality = SamlBase.c_cardinality.copy()
    c_children['{urn:oasis:names:tc:SAML:2.0:ac:classes:InternetProtocolPassword}GoverningAgreementRef'] = ('governing_agreement_ref', [GoverningAgreementRef])
    c_cardinality['governing_agreement_ref'] = {'min': 1}
    c_child_order.extend(['governing_agreement_ref'])

    def __init__(self, governing_agreement_ref=None, text=None, extension_elements=None, extension_attributes=None):
        SamlBase.__init__(self, text=text, extension_elements=extension_elements, extension_attributes=extension_attributes)
        self.governing_agreement_ref = governing_agreement_ref or []