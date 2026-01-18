import saml2
from saml2 import SamlBase
from saml2 import md
class PublicationPath(PublicationPathType_):
    """The urn:oasis:names:tc:SAML:metadata:rpi:PublicationPath element"""
    c_tag = 'PublicationPath'
    c_namespace = NAMESPACE
    c_children = PublicationPathType_.c_children.copy()
    c_attributes = PublicationPathType_.c_attributes.copy()
    c_child_order = PublicationPathType_.c_child_order[:]
    c_cardinality = PublicationPathType_.c_cardinality.copy()