import saml2
from saml2 import SamlBase
from saml2.ws import wsaddr as wsa
from saml2.ws import wssec as wsse
from saml2.ws import wsutil as wsu
class RequestSecurityTokenResponseCollectionType_(SamlBase):
    """The http://docs.oasis-open.org/ws-sx/ws-trust/200512/:RequestSecurityTokenResponseCollectionType element"""
    c_tag = 'RequestSecurityTokenResponseCollectionType'
    c_namespace = NAMESPACE
    c_children = SamlBase.c_children.copy()
    c_attributes = SamlBase.c_attributes.copy()
    c_child_order = SamlBase.c_child_order[:]
    c_cardinality = SamlBase.c_cardinality.copy()
    c_children['{http://docs.oasis-open.org/ws-sx/ws-trust/200512/}RequestSecurityTokenResponse'] = ('request_security_token_response', [RequestSecurityTokenResponse])
    c_cardinality['request_security_token_response'] = {'min': 1}
    c_child_order.extend(['request_security_token_response'])

    def __init__(self, request_security_token_response=None, text=None, extension_elements=None, extension_attributes=None):
        SamlBase.__init__(self, text=text, extension_elements=extension_elements, extension_attributes=extension_attributes)
        self.request_security_token_response = request_security_token_response or []