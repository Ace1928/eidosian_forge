import saml2
from saml2 import SamlBase
from saml2 import saml
from saml2 import xmldsig as ds
class AuthzDecisionQueryType_(SubjectQueryAbstractType_):
    """The urn:oasis:names:tc:SAML:2.0:protocol:AuthzDecisionQueryType
    element"""
    c_tag = 'AuthzDecisionQueryType'
    c_namespace = NAMESPACE
    c_children = SubjectQueryAbstractType_.c_children.copy()
    c_attributes = SubjectQueryAbstractType_.c_attributes.copy()
    c_child_order = SubjectQueryAbstractType_.c_child_order[:]
    c_cardinality = SubjectQueryAbstractType_.c_cardinality.copy()
    c_children['{urn:oasis:names:tc:SAML:2.0:assertion}Action'] = ('action', [saml.Action])
    c_cardinality['action'] = {'min': 1}
    c_children['{urn:oasis:names:tc:SAML:2.0:assertion}Evidence'] = ('evidence', saml.Evidence)
    c_cardinality['evidence'] = {'min': 0, 'max': 1}
    c_attributes['Resource'] = ('resource', 'anyURI', True)
    c_child_order.extend(['action', 'evidence'])

    def __init__(self, action=None, evidence=None, resource=None, subject=None, issuer=None, signature=None, extensions=None, id=None, version=None, issue_instant=None, destination=None, consent=None, text=None, extension_elements=None, extension_attributes=None):
        SubjectQueryAbstractType_.__init__(self, subject=subject, issuer=issuer, signature=signature, extensions=extensions, id=id, version=version, issue_instant=issue_instant, destination=destination, consent=consent, text=text, extension_elements=extension_elements, extension_attributes=extension_attributes)
        self.action = action or []
        self.evidence = evidence
        self.resource = resource