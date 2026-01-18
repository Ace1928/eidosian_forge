import saml2
from saml2 import SamlBase
class KeyValueType_(SamlBase):
    """The http://www.w3.org/2000/09/xmldsig#:KeyValueType element"""
    c_tag = 'KeyValueType'
    c_namespace = NAMESPACE
    c_children = SamlBase.c_children.copy()
    c_attributes = SamlBase.c_attributes.copy()
    c_child_order = SamlBase.c_child_order[:]
    c_cardinality = SamlBase.c_cardinality.copy()
    c_children['{http://www.w3.org/2000/09/xmldsig#}DSAKeyValue'] = ('dsa_key_value', DSAKeyValue)
    c_cardinality['dsa_key_value'] = {'min': 0, 'max': 1}
    c_children['{http://www.w3.org/2000/09/xmldsig#}RSAKeyValue'] = ('rsa_key_value', RSAKeyValue)
    c_cardinality['rsa_key_value'] = {'min': 0, 'max': 1}
    c_child_order.extend(['dsa_key_value', 'rsa_key_value'])

    def __init__(self, dsa_key_value=None, rsa_key_value=None, text=None, extension_elements=None, extension_attributes=None):
        SamlBase.__init__(self, text=text, extension_elements=extension_elements, extension_attributes=extension_attributes)
        self.dsa_key_value = dsa_key_value
        self.rsa_key_value = rsa_key_value