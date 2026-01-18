import saml2
from saml2 import SamlBase
from saml2 import md
class InformationURL(md.LocalizedURIType_):
    """The urn:oasis:names:tc:SAML:metadata:ui:InformationURL element"""
    c_tag = 'InformationURL'
    c_namespace = NAMESPACE
    c_children = md.LocalizedURIType_.c_children.copy()
    c_attributes = md.LocalizedURIType_.c_attributes.copy()
    c_child_order = md.LocalizedURIType_.c_child_order[:]
    c_cardinality = md.LocalizedURIType_.c_cardinality.copy()