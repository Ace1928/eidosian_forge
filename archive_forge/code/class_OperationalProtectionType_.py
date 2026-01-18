import saml2
from saml2 import SamlBase
class OperationalProtectionType_(SamlBase):
    """The urn:oasis:names:tc:SAML:2.0:ac:classes:InternetProtocolPassword:OperationalProtectionType element"""
    c_tag = 'OperationalProtectionType'
    c_namespace = NAMESPACE
    c_children = SamlBase.c_children.copy()
    c_attributes = SamlBase.c_attributes.copy()
    c_child_order = SamlBase.c_child_order[:]
    c_cardinality = SamlBase.c_cardinality.copy()
    c_children['{urn:oasis:names:tc:SAML:2.0:ac:classes:InternetProtocolPassword}SecurityAudit'] = ('security_audit', SecurityAudit)
    c_cardinality['security_audit'] = {'min': 0, 'max': 1}
    c_children['{urn:oasis:names:tc:SAML:2.0:ac:classes:InternetProtocolPassword}DeactivationCallCenter'] = ('deactivation_call_center', DeactivationCallCenter)
    c_cardinality['deactivation_call_center'] = {'min': 0, 'max': 1}
    c_children['{urn:oasis:names:tc:SAML:2.0:ac:classes:InternetProtocolPassword}Extension'] = ('extension', [Extension])
    c_cardinality['extension'] = {'min': 0}
    c_child_order.extend(['security_audit', 'deactivation_call_center', 'extension'])

    def __init__(self, security_audit=None, deactivation_call_center=None, extension=None, text=None, extension_elements=None, extension_attributes=None):
        SamlBase.__init__(self, text=text, extension_elements=extension_elements, extension_attributes=extension_attributes)
        self.security_audit = security_audit
        self.deactivation_call_center = deactivation_call_center
        self.extension = extension or []