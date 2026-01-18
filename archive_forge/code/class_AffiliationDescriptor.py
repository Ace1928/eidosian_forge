import saml2
from saml2 import SamlBase
from saml2 import saml
from saml2 import xmldsig as ds
from saml2 import xmlenc as xenc
class AffiliationDescriptor(AffiliationDescriptorType_):
    """The urn:oasis:names:tc:SAML:2.0:metadata:AffiliationDescriptor element"""
    c_tag = 'AffiliationDescriptor'
    c_namespace = NAMESPACE
    c_children = AffiliationDescriptorType_.c_children.copy()
    c_attributes = AffiliationDescriptorType_.c_attributes.copy()
    c_child_order = AffiliationDescriptorType_.c_child_order[:]
    c_cardinality = AffiliationDescriptorType_.c_cardinality.copy()