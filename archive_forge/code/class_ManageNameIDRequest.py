import saml2
from saml2 import SamlBase
from saml2 import saml
from saml2 import xmldsig as ds
class ManageNameIDRequest(ManageNameIDRequestType_):
    """The urn:oasis:names:tc:SAML:2.0:protocol:ManageNameIDRequest element"""
    c_tag = 'ManageNameIDRequest'
    c_namespace = NAMESPACE
    c_children = ManageNameIDRequestType_.c_children.copy()
    c_attributes = ManageNameIDRequestType_.c_attributes.copy()
    c_child_order = ManageNameIDRequestType_.c_child_order[:]
    c_cardinality = ManageNameIDRequestType_.c_cardinality.copy()