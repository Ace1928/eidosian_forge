import saml2
from saml2 import SamlBase
from saml2 import md
class DiscoHints(DiscoHintsType_):
    """The urn:oasis:names:tc:SAML:metadata:ui:DiscoHints element"""
    c_tag = 'DiscoHints'
    c_namespace = NAMESPACE
    c_children = DiscoHintsType_.c_children.copy()
    c_attributes = DiscoHintsType_.c_attributes.copy()
    c_child_order = DiscoHintsType_.c_child_order[:]
    c_cardinality = DiscoHintsType_.c_cardinality.copy()