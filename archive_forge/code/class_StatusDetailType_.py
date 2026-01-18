import saml2
from saml2 import SamlBase
from saml2 import saml
from saml2 import xmldsig as ds
class StatusDetailType_(SamlBase):
    """The urn:oasis:names:tc:SAML:2.0:protocol:StatusDetailType element"""
    c_tag = 'StatusDetailType'
    c_namespace = NAMESPACE
    c_children = SamlBase.c_children.copy()
    c_attributes = SamlBase.c_attributes.copy()
    c_child_order = SamlBase.c_child_order[:]
    c_cardinality = SamlBase.c_cardinality.copy()
    c_any = {'namespace': '##any', 'processContents': 'lax', 'minOccurs': '0', 'maxOccurs': 'unbounded'}