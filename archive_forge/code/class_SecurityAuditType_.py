import saml2
from saml2 import SamlBase
class SecurityAuditType_(SamlBase):
    """The urn:oasis:names:tc:SAML:2.0:ac:classes:InternetProtocolPassword:SecurityAuditType element"""
    c_tag = 'SecurityAuditType'
    c_namespace = NAMESPACE
    c_children = SamlBase.c_children.copy()
    c_attributes = SamlBase.c_attributes.copy()
    c_child_order = SamlBase.c_child_order[:]
    c_cardinality = SamlBase.c_cardinality.copy()
    c_children['{urn:oasis:names:tc:SAML:2.0:ac:classes:InternetProtocolPassword}SwitchAudit'] = ('switch_audit', SwitchAudit)
    c_cardinality['switch_audit'] = {'min': 0, 'max': 1}
    c_children['{urn:oasis:names:tc:SAML:2.0:ac:classes:InternetProtocolPassword}Extension'] = ('extension', [Extension])
    c_cardinality['extension'] = {'min': 0}
    c_child_order.extend(['switch_audit', 'extension'])

    def __init__(self, switch_audit=None, extension=None, text=None, extension_elements=None, extension_attributes=None):
        SamlBase.__init__(self, text=text, extension_elements=extension_elements, extension_attributes=extension_attributes)
        self.switch_audit = switch_audit
        self.extension = extension or []