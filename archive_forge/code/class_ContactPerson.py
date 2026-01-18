import saml2
from saml2 import SamlBase
from saml2 import saml
from saml2 import xmldsig as ds
from saml2 import xmlenc as xenc
class ContactPerson(ContactType_):
    """The urn:oasis:names:tc:SAML:2.0:metadata:ContactPerson element"""
    c_tag = 'ContactPerson'
    c_namespace = NAMESPACE
    c_children = ContactType_.c_children.copy()
    c_attributes = ContactType_.c_attributes.copy()
    c_child_order = ContactType_.c_child_order[:]
    c_cardinality = ContactType_.c_cardinality.copy()