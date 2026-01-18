import saml2
from saml2 import SamlBase
from saml2.ws import wsaddr as wsa
from saml2.ws import wssec as wsse
from saml2.ws import wsutil as wsu
class AuthenticatorType_(SamlBase):
    """The http://docs.oasis-open.org/ws-sx/ws-trust/200512/:AuthenticatorType element"""
    c_tag = 'AuthenticatorType'
    c_namespace = NAMESPACE
    c_children = SamlBase.c_children.copy()
    c_attributes = SamlBase.c_attributes.copy()
    c_child_order = SamlBase.c_child_order[:]
    c_cardinality = SamlBase.c_cardinality.copy()
    c_children['{http://docs.oasis-open.org/ws-sx/ws-trust/200512/}CombinedHash'] = ('combined_hash', CombinedHash)
    c_cardinality['combined_hash'] = {'min': 0, 'max': 1}
    c_child_order.extend(['combined_hash'])

    def __init__(self, combined_hash=None, text=None, extension_elements=None, extension_attributes=None):
        SamlBase.__init__(self, text=text, extension_elements=extension_elements, extension_attributes=extension_attributes)
        self.combined_hash = combined_hash