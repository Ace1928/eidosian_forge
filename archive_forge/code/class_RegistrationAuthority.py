import saml2
from saml2 import SamlBase
from saml2 import md
class RegistrationAuthority(md.EntityIDType_):
    """The urn:oasis:names:tc:SAML:2.0:metadata:dri:RegistrationAuthority
    element"""
    c_tag = 'RegistrationAuthority'
    c_namespace = NAMESPACE
    c_children = md.EntityIDType_.c_children.copy()
    c_attributes = md.EntityIDType_.c_attributes.copy()
    c_child_order = md.EntityIDType_.c_child_order[:]
    c_cardinality = md.EntityIDType_.c_cardinality.copy()