import saml2
from saml2 import SamlBase
class SecurityAudit(SecurityAuditType_):
    """The urn:oasis:names:tc:SAML:2.0:ac:classes:InternetProtocolPassword:SecurityAudit element"""
    c_tag = 'SecurityAudit'
    c_namespace = NAMESPACE
    c_children = SecurityAuditType_.c_children.copy()
    c_attributes = SecurityAuditType_.c_attributes.copy()
    c_child_order = SecurityAuditType_.c_child_order[:]
    c_cardinality = SecurityAuditType_.c_cardinality.copy()