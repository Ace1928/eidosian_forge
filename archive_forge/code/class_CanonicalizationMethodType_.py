import saml2
from saml2 import SamlBase
class CanonicalizationMethodType_(SamlBase):
    """The http://www.w3.org/2000/09/xmldsig#:CanonicalizationMethodType
    element"""
    c_tag = 'CanonicalizationMethodType'
    c_namespace = NAMESPACE
    c_children = SamlBase.c_children.copy()
    c_attributes = SamlBase.c_attributes.copy()
    c_child_order = SamlBase.c_child_order[:]
    c_cardinality = SamlBase.c_cardinality.copy()
    c_attributes['Algorithm'] = ('algorithm', 'anyURI', True)

    def __init__(self, algorithm=None, text=None, extension_elements=None, extension_attributes=None):
        SamlBase.__init__(self, text=text, extension_elements=extension_elements, extension_attributes=extension_attributes)
        self.algorithm = algorithm