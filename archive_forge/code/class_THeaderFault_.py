import saml2
from saml2 import SamlBase
from saml2.schema import wsdl
class THeaderFault_(SamlBase):
    """The http://schemas.xmlsoap.org/wsdl/soap/:tHeaderFault element"""
    c_tag = 'tHeaderFault'
    c_namespace = NAMESPACE
    c_children = SamlBase.c_children.copy()
    c_attributes = SamlBase.c_attributes.copy()
    c_child_order = SamlBase.c_child_order[:]
    c_cardinality = SamlBase.c_cardinality.copy()
    c_attributes['message'] = ('message', 'QName', True)
    c_attributes['part'] = ('part', 'NMTOKEN', True)
    c_attributes['use'] = ('use', UseChoice_, True)
    c_attributes['encodingStyle'] = ('encoding_style', EncodingStyle_, False)
    c_attributes['namespace'] = ('namespace', 'anyURI', False)

    def __init__(self, message=None, part=None, use=None, encoding_style=None, namespace=None, text=None, extension_elements=None, extension_attributes=None):
        SamlBase.__init__(self, text=text, extension_elements=extension_elements, extension_attributes=extension_attributes)
        self.message = message
        self.part = part
        self.use = use
        self.encoding_style = encoding_style
        self.namespace = namespace