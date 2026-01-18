import saml2
from saml2 import SamlBase
from saml2 import xmldsig as ds
class EncryptionMethodType_(SamlBase):
    """The http://www.w3.org/2001/04/xmlenc#:EncryptionMethodType element"""
    c_tag = 'EncryptionMethodType'
    c_namespace = NAMESPACE
    c_children = SamlBase.c_children.copy()
    c_attributes = SamlBase.c_attributes.copy()
    c_child_order = SamlBase.c_child_order[:]
    c_cardinality = SamlBase.c_cardinality.copy()
    c_children['{http://www.w3.org/2001/04/xmlenc#}KeySize'] = ('key_size', KeySize)
    c_cardinality['key_size'] = {'min': 0, 'max': 1}
    c_children['{http://www.w3.org/2001/04/xmlenc#}OAEPparams'] = ('oae_pparams', OAEPparams)
    c_cardinality['oae_pparams'] = {'min': 0, 'max': 1}
    c_attributes['Algorithm'] = ('algorithm', 'anyURI', True)
    c_child_order.extend(['key_size', 'oae_pparams'])

    def __init__(self, key_size=None, oae_pparams=None, algorithm=None, text=None, extension_elements=None, extension_attributes=None):
        SamlBase.__init__(self, text=text, extension_elements=extension_elements, extension_attributes=extension_attributes)
        self.key_size = key_size
        self.oae_pparams = oae_pparams
        self.algorithm = algorithm