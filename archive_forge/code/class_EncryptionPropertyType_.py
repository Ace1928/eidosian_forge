import saml2
from saml2 import SamlBase
from saml2 import xmldsig as ds
class EncryptionPropertyType_(SamlBase):
    """The http://www.w3.org/2001/04/xmlenc#:EncryptionPropertyType element"""
    c_tag = 'EncryptionPropertyType'
    c_namespace = NAMESPACE
    c_children = SamlBase.c_children.copy()
    c_attributes = SamlBase.c_attributes.copy()
    c_child_order = SamlBase.c_child_order[:]
    c_cardinality = SamlBase.c_cardinality.copy()
    c_attributes['Target'] = ('target', 'anyURI', False)
    c_attributes['Id'] = ('id', 'ID', False)

    def __init__(self, target=None, id=None, text=None, extension_elements=None, extension_attributes=None):
        SamlBase.__init__(self, text=text, extension_elements=extension_elements, extension_attributes=extension_attributes)
        self.target = target
        self.id = id