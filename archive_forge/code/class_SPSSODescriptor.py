import saml2
from saml2 import SamlBase
from saml2 import saml
from saml2 import xmldsig as ds
from saml2 import xmlenc as xenc
class SPSSODescriptor(SPSSODescriptorType_):
    """The urn:oasis:names:tc:SAML:2.0:metadata:SPSSODescriptor element"""
    c_tag = 'SPSSODescriptor'
    c_namespace = NAMESPACE
    c_children = SPSSODescriptorType_.c_children.copy()
    c_attributes = SPSSODescriptorType_.c_attributes.copy()
    c_child_order = SPSSODescriptorType_.c_child_order[:]
    c_cardinality = SPSSODescriptorType_.c_cardinality.copy()