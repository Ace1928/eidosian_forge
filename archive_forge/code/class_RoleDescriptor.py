import saml2
from saml2 import SamlBase
from saml2 import saml
from saml2 import xmldsig as ds
from saml2 import xmlenc as xenc
class RoleDescriptor(RoleDescriptorType_):
    """The urn:oasis:names:tc:SAML:2.0:metadata:RoleDescriptor element"""
    c_tag = 'RoleDescriptor'
    c_namespace = NAMESPACE
    c_children = RoleDescriptorType_.c_children.copy()
    c_attributes = RoleDescriptorType_.c_attributes.copy()
    c_child_order = RoleDescriptorType_.c_child_order[:]
    c_cardinality = RoleDescriptorType_.c_cardinality.copy()