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
class AttributeStatementType_(StatementAbstractType_):
    """The urn:oasis:names:tc:SAML:2.0:assertion:AttributeStatementType
    element"""
    c_tag = 'AttributeStatementType'
    c_namespace = NAMESPACE
    c_children = StatementAbstractType_.c_children.copy()
    c_attributes = StatementAbstractType_.c_attributes.copy()
    c_child_order = StatementAbstractType_.c_child_order[:]
    c_cardinality = StatementAbstractType_.c_cardinality.copy()
    c_children['{urn:oasis:names:tc:SAML:2.0:assertion}Attribute'] = ('attribute', [Attribute])
    c_cardinality['attribute'] = {'min': 0}
    c_children['{urn:oasis:names:tc:SAML:2.0:assertion}EncryptedAttribute'] = ('encrypted_attribute', [EncryptedAttribute])
    c_cardinality['encrypted_attribute'] = {'min': 0}
    c_child_order.extend(['attribute', 'encrypted_attribute'])

    def __init__(self, attribute=None, encrypted_attribute=None, text=None, extension_elements=None, extension_attributes=None):
        StatementAbstractType_.__init__(self, text=text, extension_elements=extension_elements, extension_attributes=extension_attributes)
        self.attribute = attribute or []
        self.encrypted_attribute = encrypted_attribute or []