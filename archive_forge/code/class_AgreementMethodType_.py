import saml2
from saml2 import SamlBase
from saml2 import xmldsig as ds
class AgreementMethodType_(SamlBase):
    """The http://www.w3.org/2001/04/xmlenc#:AgreementMethodType element"""
    c_tag = 'AgreementMethodType'
    c_namespace = NAMESPACE
    c_children = SamlBase.c_children.copy()
    c_attributes = SamlBase.c_attributes.copy()
    c_child_order = SamlBase.c_child_order[:]
    c_cardinality = SamlBase.c_cardinality.copy()
    c_children['{http://www.w3.org/2001/04/xmlenc#}KA_Nonce'] = ('k_a__nonce', KA_Nonce)
    c_cardinality['k_a__nonce'] = {'min': 0, 'max': 1}
    c_children['{http://www.w3.org/2001/04/xmlenc#}OriginatorKeyInfo'] = ('originator_key_info', OriginatorKeyInfo)
    c_cardinality['originator_key_info'] = {'min': 0, 'max': 1}
    c_children['{http://www.w3.org/2001/04/xmlenc#}RecipientKeyInfo'] = ('recipient_key_info', RecipientKeyInfo)
    c_cardinality['recipient_key_info'] = {'min': 0, 'max': 1}
    c_attributes['Algorithm'] = ('algorithm', 'anyURI', True)
    c_child_order.extend(['k_a__nonce', 'originator_key_info', 'recipient_key_info'])

    def __init__(self, k_a__nonce=None, originator_key_info=None, recipient_key_info=None, algorithm=None, text=None, extension_elements=None, extension_attributes=None):
        SamlBase.__init__(self, text=text, extension_elements=extension_elements, extension_attributes=extension_attributes)
        self.k_a__nonce = k_a__nonce
        self.originator_key_info = originator_key_info
        self.recipient_key_info = recipient_key_info
        self.algorithm = algorithm