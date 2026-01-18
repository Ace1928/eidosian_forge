import saml2
from saml2 import SamlBase
from saml2 import saml
from saml2 import xmldsig as ds
from saml2 import xmlenc as xenc
class IDPSSODescriptor(IDPSSODescriptorType_):
    """The urn:oasis:names:tc:SAML:2.0:metadata:IDPSSODescriptor element"""
    c_tag = 'IDPSSODescriptor'
    c_namespace = NAMESPACE
    c_children = IDPSSODescriptorType_.c_children.copy()
    c_attributes = IDPSSODescriptorType_.c_attributes.copy()
    c_child_order = IDPSSODescriptorType_.c_child_order[:]
    c_cardinality = IDPSSODescriptorType_.c_cardinality.copy()