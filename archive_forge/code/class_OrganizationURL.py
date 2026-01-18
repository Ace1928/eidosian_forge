import saml2
from saml2 import SamlBase
from saml2 import saml
from saml2 import xmldsig as ds
from saml2 import xmlenc as xenc
class OrganizationURL(LocalizedURIType_):
    """The urn:oasis:names:tc:SAML:2.0:metadata:OrganizationURL element"""
    c_tag = 'OrganizationURL'
    c_namespace = NAMESPACE
    c_children = LocalizedURIType_.c_children.copy()
    c_attributes = LocalizedURIType_.c_attributes.copy()
    c_child_order = LocalizedURIType_.c_child_order[:]
    c_cardinality = LocalizedURIType_.c_cardinality.copy()