import saml2
from saml2 import SamlBase
from saml2 import md
class PrivacyStatementURL(md.LocalizedURIType_):
    """The urn:oasis:names:tc:SAML:metadata:ui:PrivacyStatementURL element"""
    c_tag = 'PrivacyStatementURL'
    c_namespace = NAMESPACE
    c_children = md.LocalizedURIType_.c_children.copy()
    c_attributes = md.LocalizedURIType_.c_attributes.copy()
    c_child_order = md.LocalizedURIType_.c_child_order[:]
    c_cardinality = md.LocalizedURIType_.c_cardinality.copy()