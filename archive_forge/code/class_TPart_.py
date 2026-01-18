import saml2
from saml2 import SamlBase
class TPart_(TExtensibleAttributesDocumented_):
    """The http://schemas.xmlsoap.org/wsdl/:tPart element"""
    c_tag = 'tPart'
    c_namespace = NAMESPACE
    c_children = TExtensibleAttributesDocumented_.c_children.copy()
    c_attributes = TExtensibleAttributesDocumented_.c_attributes.copy()
    c_child_order = TExtensibleAttributesDocumented_.c_child_order[:]
    c_cardinality = TExtensibleAttributesDocumented_.c_cardinality.copy()
    c_attributes['name'] = ('name', 'NCName', True)
    c_attributes['element'] = ('element', 'QName', False)
    c_attributes['type'] = ('type', 'QName', False)

    def __init__(self, name=None, element=None, type=None, documentation=None, text=None, extension_elements=None, extension_attributes=None):
        TExtensibleAttributesDocumented_.__init__(self, documentation=documentation, text=text, extension_elements=extension_elements, extension_attributes=extension_attributes)
        self.name = name
        self.element = element
        self.type = type