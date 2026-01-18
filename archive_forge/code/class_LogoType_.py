import saml2
from saml2 import SamlBase
from saml2 import md
class LogoType_(SamlBase):
    """The urn:oasis:names:tc:SAML:metadata:ui:LogoType element"""
    c_tag = 'LogoType'
    c_namespace = NAMESPACE
    c_value_type = {'base': 'anyURI'}
    c_children = SamlBase.c_children.copy()
    c_attributes = SamlBase.c_attributes.copy()
    c_child_order = SamlBase.c_child_order[:]
    c_cardinality = SamlBase.c_cardinality.copy()
    c_attributes['height'] = ('height', 'positiveInteger', True)
    c_attributes['width'] = ('width', 'positiveInteger', True)
    c_attributes['{http://www.w3.org/XML/1998/namespace}lang'] = ('lang', 'anyURI', False)

    def __init__(self, height=None, width=None, lang=None, text=None, extension_elements=None, extension_attributes=None):
        SamlBase.__init__(self, text=text, extension_elements=extension_elements, extension_attributes=extension_attributes)
        self.height = height
        self.width = width
        self.lang = lang