import saml2
from saml2 import SamlBase
from saml2 import md
class UIInfoType_(SamlBase):
    """The urn:oasis:names:tc:SAML:metadata:ui:UIInfoType element"""
    c_tag = 'UIInfoType'
    c_namespace = NAMESPACE
    c_children = SamlBase.c_children.copy()
    c_attributes = SamlBase.c_attributes.copy()
    c_child_order = SamlBase.c_child_order[:]
    c_cardinality = SamlBase.c_cardinality.copy()
    c_children['{urn:oasis:names:tc:SAML:metadata:ui}DisplayName'] = ('display_name', [DisplayName])
    c_cardinality['display_name'] = {'min': 0}
    c_children['{urn:oasis:names:tc:SAML:metadata:ui}Description'] = ('description', [Description])
    c_cardinality['description'] = {'min': 0}
    c_children['{urn:oasis:names:tc:SAML:metadata:ui}Keywords'] = ('keywords', [Keywords])
    c_cardinality['keywords'] = {'min': 0}
    c_children['{urn:oasis:names:tc:SAML:metadata:ui}Logo'] = ('logo', [Logo])
    c_cardinality['logo'] = {'min': 0}
    c_children['{urn:oasis:names:tc:SAML:metadata:ui}InformationURL'] = ('information_url', [InformationURL])
    c_cardinality['information_url'] = {'min': 0}
    c_children['{urn:oasis:names:tc:SAML:metadata:ui}PrivacyStatementURL'] = ('privacy_statement_url', [PrivacyStatementURL])
    c_cardinality['privacy_statement_url'] = {'min': 0}
    c_child_order.extend(['display_name', 'description', 'keywords', 'logo', 'information_url', 'privacy_statement_url'])

    def __init__(self, display_name=None, description=None, keywords=None, logo=None, information_url=None, privacy_statement_url=None, text=None, extension_elements=None, extension_attributes=None):
        SamlBase.__init__(self, text=text, extension_elements=extension_elements, extension_attributes=extension_attributes)
        self.display_name = display_name or []
        self.description = description or []
        self.keywords = keywords or []
        self.logo = logo or []
        self.information_url = information_url or []
        self.privacy_statement_url = privacy_statement_url or []