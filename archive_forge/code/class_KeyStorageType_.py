import saml2
from saml2 import SamlBase
class KeyStorageType_(SamlBase):
    """The urn:oasis:names:tc:SAML:2.0:ac:classes:InternetProtocolPassword:KeyStorageType element"""
    c_tag = 'KeyStorageType'
    c_namespace = NAMESPACE
    c_children = SamlBase.c_children.copy()
    c_attributes = SamlBase.c_attributes.copy()
    c_child_order = SamlBase.c_child_order[:]
    c_cardinality = SamlBase.c_cardinality.copy()
    c_attributes['medium'] = ('medium', MediumType_, True)

    def __init__(self, medium=None, text=None, extension_elements=None, extension_attributes=None):
        SamlBase.__init__(self, text=text, extension_elements=extension_elements, extension_attributes=extension_attributes)
        self.medium = medium