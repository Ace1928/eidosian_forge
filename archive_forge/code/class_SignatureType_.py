import saml2
from saml2 import SamlBase
class SignatureType_(SamlBase):
    """The http://www.w3.org/2000/09/xmldsig#:SignatureType element"""
    c_tag = 'SignatureType'
    c_namespace = NAMESPACE
    c_children = SamlBase.c_children.copy()
    c_attributes = SamlBase.c_attributes.copy()
    c_child_order = SamlBase.c_child_order[:]
    c_cardinality = SamlBase.c_cardinality.copy()
    c_children['{http://www.w3.org/2000/09/xmldsig#}SignedInfo'] = ('signed_info', SignedInfo)
    c_children['{http://www.w3.org/2000/09/xmldsig#}SignatureValue'] = ('signature_value', SignatureValue)
    c_children['{http://www.w3.org/2000/09/xmldsig#}KeyInfo'] = ('key_info', KeyInfo)
    c_cardinality['key_info'] = {'min': 0, 'max': 1}
    c_children['{http://www.w3.org/2000/09/xmldsig#}Object'] = ('object', [Object])
    c_cardinality['object'] = {'min': 0}
    c_attributes['Id'] = ('id', 'ID', False)
    c_child_order.extend(['signed_info', 'signature_value', 'key_info', 'object'])

    def __init__(self, signed_info=None, signature_value=None, key_info=None, object=None, id=None, text=None, extension_elements=None, extension_attributes=None):
        SamlBase.__init__(self, text=text, extension_elements=extension_elements, extension_attributes=extension_attributes)
        self.signed_info = signed_info
        self.signature_value = signature_value
        self.key_info = key_info
        self.object = object or []
        self.id = id