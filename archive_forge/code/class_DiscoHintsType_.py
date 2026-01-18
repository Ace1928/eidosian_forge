import saml2
from saml2 import SamlBase
from saml2 import md
class DiscoHintsType_(SamlBase):
    """The urn:oasis:names:tc:SAML:metadata:ui:DiscoHintsType element"""
    c_tag = 'DiscoHintsType'
    c_namespace = NAMESPACE
    c_children = SamlBase.c_children.copy()
    c_attributes = SamlBase.c_attributes.copy()
    c_child_order = SamlBase.c_child_order[:]
    c_cardinality = SamlBase.c_cardinality.copy()
    c_children['{urn:oasis:names:tc:SAML:metadata:ui}IPHint'] = ('ip_hint', [IPHint])
    c_cardinality['ip_hint'] = {'min': 0}
    c_children['{urn:oasis:names:tc:SAML:metadata:ui}DomainHint'] = ('domain_hint', [DomainHint])
    c_cardinality['domain_hint'] = {'min': 0}
    c_children['{urn:oasis:names:tc:SAML:metadata:ui}GeolocationHint'] = ('geolocation_hint', [GeolocationHint])
    c_cardinality['geolocation_hint'] = {'min': 0}
    c_child_order.extend(['ip_hint', 'domain_hint', 'geolocation_hint'])

    def __init__(self, ip_hint=None, domain_hint=None, geolocation_hint=None, text=None, extension_elements=None, extension_attributes=None):
        SamlBase.__init__(self, text=text, extension_elements=extension_elements, extension_attributes=extension_attributes)
        self.ip_hint = ip_hint or []
        self.domain_hint = domain_hint or []
        self.geolocation_hint = geolocation_hint or []