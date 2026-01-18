import saml2
from saml2 import SamlBase
from saml2 import saml
from saml2 import xmldsig as ds
class RequestedAuthnContextType_(SamlBase):
    """The urn:oasis:names:tc:SAML:2.0:protocol:RequestedAuthnContextType
    element"""
    c_tag = 'RequestedAuthnContextType'
    c_namespace = NAMESPACE
    c_children = SamlBase.c_children.copy()
    c_attributes = SamlBase.c_attributes.copy()
    c_child_order = SamlBase.c_child_order[:]
    c_cardinality = SamlBase.c_cardinality.copy()
    c_children['{urn:oasis:names:tc:SAML:2.0:assertion}AuthnContextClassRef'] = ('authn_context_class_ref', [saml.AuthnContextClassRef])
    c_cardinality['authn_context_class_ref'] = {'min': 0}
    c_children['{urn:oasis:names:tc:SAML:2.0:assertion}AuthnContextDeclRef'] = ('authn_context_decl_ref', [saml.AuthnContextDeclRef])
    c_cardinality['authn_context_decl_ref'] = {'min': 0}
    c_attributes['Comparison'] = ('comparison', AuthnContextComparisonType_, False)
    c_child_order.extend(['authn_context_class_ref', 'authn_context_decl_ref'])

    def __init__(self, authn_context_class_ref=None, authn_context_decl_ref=None, comparison=None, text=None, extension_elements=None, extension_attributes=None):
        SamlBase.__init__(self, text=text, extension_elements=extension_elements, extension_attributes=extension_attributes)
        self.authn_context_class_ref = authn_context_class_ref or []
        self.authn_context_decl_ref = authn_context_decl_ref or []
        self.comparison = comparison