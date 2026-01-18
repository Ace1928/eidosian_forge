import saml2
from saml2 import SamlBase
from saml2 import md
class Publication(PublicationType_):
    """The urn:oasis:names:tc:SAML:metadata:rpi:Publication element"""
    c_tag = 'Publication'
    c_namespace = NAMESPACE
    c_children = PublicationType_.c_children.copy()
    c_attributes = PublicationType_.c_attributes.copy()
    c_child_order = PublicationType_.c_child_order[:]
    c_cardinality = PublicationType_.c_cardinality.copy()