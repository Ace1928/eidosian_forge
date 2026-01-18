import saml2
from saml2 import SamlBase
from saml2.ws import wsaddr as wsa
from saml2.ws import wssec as wsse
from saml2.ws import wsutil as wsu
class SignChallengeType_(SamlBase):
    """The http://docs.oasis-open.org/ws-sx/ws-trust/200512/:SignChallengeType element"""
    c_tag = 'SignChallengeType'
    c_namespace = NAMESPACE
    c_children = SamlBase.c_children.copy()
    c_attributes = SamlBase.c_attributes.copy()
    c_child_order = SamlBase.c_child_order[:]
    c_cardinality = SamlBase.c_cardinality.copy()
    c_children['{http://docs.oasis-open.org/ws-sx/ws-trust/200512/}Challenge'] = ('challenge', Challenge)
    c_child_order.extend(['challenge'])

    def __init__(self, challenge=None, text=None, extension_elements=None, extension_attributes=None):
        SamlBase.__init__(self, text=text, extension_elements=extension_elements, extension_attributes=extension_attributes)
        self.challenge = challenge