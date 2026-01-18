import saml2
from saml2 import SamlBase
from saml2 import saml
from saml2 import xmldsig as ds
from saml2 import xmlenc as xenc
class ContactType_(SamlBase):
    """The urn:oasis:names:tc:SAML:2.0:metadata:ContactType element"""
    c_tag = 'ContactType'
    c_namespace = NAMESPACE
    c_children = SamlBase.c_children.copy()
    c_attributes = SamlBase.c_attributes.copy()
    c_child_order = SamlBase.c_child_order[:]
    c_cardinality = SamlBase.c_cardinality.copy()
    c_children['{urn:oasis:names:tc:SAML:2.0:metadata}Extensions'] = ('extensions', Extensions)
    c_cardinality['extensions'] = {'min': 0, 'max': 1}
    c_children['{urn:oasis:names:tc:SAML:2.0:metadata}Company'] = ('company', Company)
    c_cardinality['company'] = {'min': 0, 'max': 1}
    c_children['{urn:oasis:names:tc:SAML:2.0:metadata}GivenName'] = ('given_name', GivenName)
    c_cardinality['given_name'] = {'min': 0, 'max': 1}
    c_children['{urn:oasis:names:tc:SAML:2.0:metadata}SurName'] = ('sur_name', SurName)
    c_cardinality['sur_name'] = {'min': 0, 'max': 1}
    c_children['{urn:oasis:names:tc:SAML:2.0:metadata}EmailAddress'] = ('email_address', [EmailAddress])
    c_cardinality['email_address'] = {'min': 0}
    c_children['{urn:oasis:names:tc:SAML:2.0:metadata}TelephoneNumber'] = ('telephone_number', [TelephoneNumber])
    c_cardinality['telephone_number'] = {'min': 0}
    c_attributes['contactType'] = ('contact_type', ContactTypeType_, True)
    c_child_order.extend(['extensions', 'company', 'given_name', 'sur_name', 'email_address', 'telephone_number'])

    def __init__(self, extensions=None, company=None, given_name=None, sur_name=None, email_address=None, telephone_number=None, contact_type=None, text=None, extension_elements=None, extension_attributes=None):
        SamlBase.__init__(self, text=text, extension_elements=extension_elements, extension_attributes=extension_attributes)
        self.extensions = extensions
        self.company = company
        self.given_name = given_name
        self.sur_name = sur_name
        self.email_address = email_address or []
        self.telephone_number = telephone_number or []
        self.contact_type = contact_type