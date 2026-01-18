import saml2
from saml2 import SamlBase
from saml2 import xmldsig as ds
class EncryptedType_(SamlBase):
    """The http://www.w3.org/2001/04/xmlenc#:EncryptedType element"""
    c_tag = 'EncryptedType'
    c_namespace = NAMESPACE
    c_children = SamlBase.c_children.copy()
    c_attributes = SamlBase.c_attributes.copy()
    c_child_order = SamlBase.c_child_order[:]
    c_cardinality = SamlBase.c_cardinality.copy()
    c_children['{http://www.w3.org/2001/04/xmlenc#}EncryptionMethod'] = ('encryption_method', EncryptionMethod)
    c_cardinality['encryption_method'] = {'min': 0, 'max': 1}
    c_children['{http://www.w3.org/2000/09/xmldsig#}KeyInfo'] = ('key_info', ds.KeyInfo)
    c_cardinality['key_info'] = {'min': 0, 'max': 1}
    c_children['{http://www.w3.org/2001/04/xmlenc#}CipherData'] = ('cipher_data', CipherData)
    c_children['{http://www.w3.org/2001/04/xmlenc#}EncryptionProperties'] = ('encryption_properties', EncryptionProperties)
    c_cardinality['encryption_properties'] = {'min': 0, 'max': 1}
    c_attributes['Id'] = ('id', 'ID', False)
    c_attributes['Type'] = ('type', 'anyURI', False)
    c_attributes['MimeType'] = ('mime_type', 'string', False)
    c_attributes['Encoding'] = ('encoding', 'anyURI', False)
    c_child_order.extend(['encryption_method', 'key_info', 'cipher_data', 'encryption_properties'])

    def __init__(self, encryption_method=None, key_info=None, cipher_data=None, encryption_properties=None, id=None, type=None, mime_type=None, encoding=None, text=None, extension_elements=None, extension_attributes=None):
        SamlBase.__init__(self, text=text, extension_elements=extension_elements, extension_attributes=extension_attributes)
        self.encryption_method = encryption_method
        self.key_info = key_info
        self.cipher_data = cipher_data
        self.encryption_properties = encryption_properties
        self.id = id
        self.type = type
        self.mime_type = mime_type
        self.encoding = encoding