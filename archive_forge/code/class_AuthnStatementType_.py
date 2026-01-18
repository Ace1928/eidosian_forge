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
class AuthnStatementType_(StatementAbstractType_):
    """The urn:oasis:names:tc:SAML:2.0:assertion:AuthnStatementType element"""
    c_tag = 'AuthnStatementType'
    c_namespace = NAMESPACE
    c_children = StatementAbstractType_.c_children.copy()
    c_attributes = StatementAbstractType_.c_attributes.copy()
    c_child_order = StatementAbstractType_.c_child_order[:]
    c_cardinality = StatementAbstractType_.c_cardinality.copy()
    c_children['{urn:oasis:names:tc:SAML:2.0:assertion}SubjectLocality'] = ('subject_locality', SubjectLocality)
    c_cardinality['subject_locality'] = {'min': 0, 'max': 1}
    c_children['{urn:oasis:names:tc:SAML:2.0:assertion}AuthnContext'] = ('authn_context', AuthnContext)
    c_attributes['AuthnInstant'] = ('authn_instant', 'dateTime', True)
    c_attributes['SessionIndex'] = ('session_index', 'string', False)
    c_attributes['SessionNotOnOrAfter'] = ('session_not_on_or_after', 'dateTime', False)
    c_child_order.extend(['subject_locality', 'authn_context'])

    def __init__(self, subject_locality=None, authn_context=None, authn_instant=None, session_index=None, session_not_on_or_after=None, text=None, extension_elements=None, extension_attributes=None):
        StatementAbstractType_.__init__(self, text=text, extension_elements=extension_elements, extension_attributes=extension_attributes)
        self.subject_locality = subject_locality
        self.authn_context = authn_context
        self.authn_instant = authn_instant
        self.session_index = session_index
        self.session_not_on_or_after = session_not_on_or_after