import base64
from datetime import date
from datetime import datetime
import saml2
from saml2 import SamlBase
from saml2 import xmldsig as ds
from saml2 import xmlenc as xenc
from saml2.validate import MustValueError
from saml2.validate import ShouldValueError
from saml2.validate import valid_domain_name
from saml2.validate import valid_ipv4
from saml2.validate import valid_ipv6
class SubjectConfirmationDataType_(SamlBase):
    """The urn:oasis:names:tc:SAML:2.0:assertion:SubjectConfirmationDataType
    element"""
    c_tag = 'SubjectConfirmationDataType'
    c_namespace = NAMESPACE
    c_children = SamlBase.c_children.copy()
    c_attributes = SamlBase.c_attributes.copy()
    c_child_order = SamlBase.c_child_order[:]
    c_cardinality = SamlBase.c_cardinality.copy()
    c_attributes['NotBefore'] = ('not_before', 'dateTime', False)
    c_attributes['NotOnOrAfter'] = ('not_on_or_after', 'dateTime', False)
    c_attributes['Recipient'] = ('recipient', 'anyURI', False)
    c_attributes['InResponseTo'] = ('in_response_to', 'NCName', False)
    c_attributes['Address'] = ('address', 'string', False)
    c_any = {'namespace': '##any', 'processContents': 'lax', 'minOccurs': '0', 'maxOccurs': 'unbounded'}
    c_any_attribute = {'namespace': '##other', 'processContents': 'lax'}

    def __init__(self, not_before=None, not_on_or_after=None, recipient=None, in_response_to=None, address=None, text=None, extension_elements=None, extension_attributes=None):
        SamlBase.__init__(self, text=text, extension_elements=extension_elements, extension_attributes=extension_attributes)
        self.not_before = not_before
        self.not_on_or_after = not_on_or_after
        self.recipient = recipient
        self.in_response_to = in_response_to
        self.address = address