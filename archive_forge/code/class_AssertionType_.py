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
class AssertionType_(SamlBase):
    """The urn:oasis:names:tc:SAML:2.0:assertion:AssertionType element"""
    c_tag = 'AssertionType'
    c_namespace = NAMESPACE
    c_children = SamlBase.c_children.copy()
    c_attributes = SamlBase.c_attributes.copy()
    c_child_order = SamlBase.c_child_order[:]
    c_cardinality = SamlBase.c_cardinality.copy()
    c_children['{urn:oasis:names:tc:SAML:2.0:assertion}Issuer'] = ('issuer', Issuer)
    c_children['{http://www.w3.org/2000/09/xmldsig#}Signature'] = ('signature', ds.Signature)
    c_cardinality['signature'] = {'min': 0, 'max': 1}
    c_children['{urn:oasis:names:tc:SAML:2.0:assertion}Subject'] = ('subject', Subject)
    c_cardinality['subject'] = {'min': 0, 'max': 1}
    c_children['{urn:oasis:names:tc:SAML:2.0:assertion}Conditions'] = ('conditions', Conditions)
    c_cardinality['conditions'] = {'min': 0, 'max': 1}
    c_cardinality['advice'] = {'min': 0, 'max': 1}
    c_children['{urn:oasis:names:tc:SAML:2.0:assertion}Statement'] = ('statement', [Statement])
    c_cardinality['statement'] = {'min': 0}
    c_children['{urn:oasis:names:tc:SAML:2.0:assertion}AuthnStatement'] = ('authn_statement', [AuthnStatement])
    c_cardinality['authn_statement'] = {'min': 0}
    c_children['{urn:oasis:names:tc:SAML:2.0:assertion}AuthzDecisionStatement'] = ('authz_decision_statement', [AuthzDecisionStatement])
    c_cardinality['authz_decision_statement'] = {'min': 0}
    c_children['{urn:oasis:names:tc:SAML:2.0:assertion}AttributeStatement'] = ('attribute_statement', [AttributeStatement])
    c_cardinality['attribute_statement'] = {'min': 0}
    c_attributes['Version'] = ('version', 'string', True)
    c_attributes['ID'] = ('id', 'ID', True)
    c_attributes['IssueInstant'] = ('issue_instant', 'dateTime', True)
    c_child_order.extend(['issuer', 'signature', 'subject', 'conditions', 'advice', 'statement', 'authn_statement', 'authz_decision_statement', 'attribute_statement'])

    def __init__(self, issuer=None, signature=None, subject=None, conditions=None, advice=None, statement=None, authn_statement=None, authz_decision_statement=None, attribute_statement=None, version=None, id=None, issue_instant=None, text=None, extension_elements=None, extension_attributes=None):
        SamlBase.__init__(self, text=text, extension_elements=extension_elements, extension_attributes=extension_attributes)
        self.issuer = issuer
        self.signature = signature
        self.subject = subject
        self.conditions = conditions
        self.advice = advice
        self.statement = statement or []
        self.authn_statement = authn_statement or []
        self.authz_decision_statement = authz_decision_statement or []
        self.attribute_statement = attribute_statement or []
        self.version = version
        self.id = id
        self.issue_instant = issue_instant

    def verify(self):
        if self.attribute_statement or self.statement or self.authn_statement or self.authz_decision_statement:
            pass
        elif not self.subject:
            raise MustValueError('If no statement MUST contain a subject element')
        if self.authn_statement and (not self.subject):
            raise MustValueError('An assertion with an AuthnStatement must contain a Subject')
        return SamlBase.verify(self)