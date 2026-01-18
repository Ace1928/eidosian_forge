import saml2
from saml2 import SamlBase
from saml2 import saml
from saml2 import xmldsig as ds
class AssertionIDRequestType_(RequestAbstractType_):
    """
    The urn:oasis:names:tc:SAML:2.0:protocol:AssertionIDRequestType element
    """
    c_tag = 'AssertionIDRequestType'
    c_namespace = NAMESPACE
    c_children = RequestAbstractType_.c_children.copy()
    c_attributes = RequestAbstractType_.c_attributes.copy()
    c_child_order = RequestAbstractType_.c_child_order[:]
    c_cardinality = RequestAbstractType_.c_cardinality.copy()
    c_children['{urn:oasis:names:tc:SAML:2.0:assertion}AssertionIDRef'] = ('assertion_id_ref', [saml.AssertionIDRef])
    c_cardinality['assertion_id_ref'] = {'min': 1}
    c_child_order.extend(['assertion_id_ref'])

    def __init__(self, assertion_id_ref=None, issuer=None, signature=None, extensions=None, id=None, version=None, issue_instant=None, destination=None, consent=None, text=None, extension_elements=None, extension_attributes=None):
        RequestAbstractType_.__init__(self, issuer=issuer, signature=signature, extensions=extensions, id=id, version=version, issue_instant=issue_instant, destination=destination, consent=consent, text=text, extension_elements=extension_elements, extension_attributes=extension_attributes)
        self.assertion_id_ref = assertion_id_ref or []