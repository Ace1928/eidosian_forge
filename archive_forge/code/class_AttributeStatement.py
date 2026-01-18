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
class AttributeStatement(AttributeStatementType_):
    """The urn:oasis:names:tc:SAML:2.0:assertion:AttributeStatement element"""
    c_tag = 'AttributeStatement'
    c_namespace = NAMESPACE
    c_children = AttributeStatementType_.c_children.copy()
    c_attributes = AttributeStatementType_.c_attributes.copy()
    c_child_order = AttributeStatementType_.c_child_order[:]
    c_cardinality = AttributeStatementType_.c_cardinality.copy()