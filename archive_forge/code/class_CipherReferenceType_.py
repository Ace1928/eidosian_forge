import saml2
from saml2 import SamlBase
from saml2 import xmldsig as ds
class CipherReferenceType_(SamlBase):
    """The http://www.w3.org/2001/04/xmlenc#:CipherReferenceType element"""
    c_tag = 'CipherReferenceType'
    c_namespace = NAMESPACE
    c_children = SamlBase.c_children.copy()
    c_attributes = SamlBase.c_attributes.copy()
    c_child_order = SamlBase.c_child_order[:]
    c_cardinality = SamlBase.c_cardinality.copy()
    c_children['{http://www.w3.org/2001/04/xmlenc#}Transforms'] = ('transforms', Transforms)
    c_cardinality['transforms'] = {'min': 0, 'max': 1}
    c_attributes['URI'] = ('uri', 'anyURI', True)
    c_child_order.extend(['transforms'])

    def __init__(self, transforms=None, uri=None, text=None, extension_elements=None, extension_attributes=None):
        SamlBase.__init__(self, text=text, extension_elements=extension_elements, extension_attributes=extension_attributes)
        self.transforms = transforms
        self.uri = uri