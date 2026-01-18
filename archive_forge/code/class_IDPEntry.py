import saml2
from saml2 import SamlBase
from saml2 import saml
from saml2 import xmldsig as ds
class IDPEntry(IDPEntryType_):
    """The urn:oasis:names:tc:SAML:2.0:protocol:IDPEntry element"""
    c_tag = 'IDPEntry'
    c_namespace = NAMESPACE
    c_children = IDPEntryType_.c_children.copy()
    c_attributes = IDPEntryType_.c_attributes.copy()
    c_child_order = IDPEntryType_.c_child_order[:]
    c_cardinality = IDPEntryType_.c_cardinality.copy()