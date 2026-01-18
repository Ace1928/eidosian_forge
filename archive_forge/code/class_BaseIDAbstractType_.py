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
class BaseIDAbstractType_(SamlBase):
    """The urn:oasis:names:tc:SAML:2.0:assertion:BaseIDAbstractType element"""
    c_tag = 'BaseIDAbstractType'
    c_namespace = NAMESPACE
    c_children = SamlBase.c_children.copy()
    c_attributes = SamlBase.c_attributes.copy()
    c_child_order = SamlBase.c_child_order[:]
    c_cardinality = SamlBase.c_cardinality.copy()
    c_attributes['NameQualifier'] = ('name_qualifier', 'string', False)
    c_attributes['SPNameQualifier'] = ('sp_name_qualifier', 'string', False)

    def __init__(self, name_qualifier=None, sp_name_qualifier=None, text=None, extension_elements=None, extension_attributes=None):
        SamlBase.__init__(self, text=text, extension_elements=extension_elements, extension_attributes=extension_attributes)
        self.name_qualifier = name_qualifier
        self.sp_name_qualifier = sp_name_qualifier