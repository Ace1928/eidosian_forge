import saml2
from saml2 import SamlBase
from saml2 import saml
from saml2 import xmldsig as ds
class NameIDMappingRequest(NameIDMappingRequestType_):
    """The urn:oasis:names:tc:SAML:2.0:protocol:NameIDMappingRequest element"""
    c_tag = 'NameIDMappingRequest'
    c_namespace = NAMESPACE
    c_children = NameIDMappingRequestType_.c_children.copy()
    c_attributes = NameIDMappingRequestType_.c_attributes.copy()
    c_child_order = NameIDMappingRequestType_.c_child_order[:]
    c_cardinality = NameIDMappingRequestType_.c_cardinality.copy()