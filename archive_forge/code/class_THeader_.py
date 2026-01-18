import saml2
from saml2 import SamlBase
from saml2.schema import wsdl
class THeader_(wsdl.TExtensibilityElement_):
    """The http://schemas.xmlsoap.org/wsdl/soap/:tHeader element"""
    c_tag = 'tHeader'
    c_namespace = NAMESPACE
    c_children = wsdl.TExtensibilityElement_.c_children.copy()
    c_attributes = wsdl.TExtensibilityElement_.c_attributes.copy()
    c_child_order = wsdl.TExtensibilityElement_.c_child_order[:]
    c_cardinality = wsdl.TExtensibilityElement_.c_cardinality.copy()
    c_children['{http://schemas.xmlsoap.org/wsdl/soap/}headerfault'] = ('headerfault', [Headerfault])
    c_cardinality['headerfault'] = {'min': 0}
    c_attributes['message'] = ('message', 'QName', True)
    c_attributes['part'] = ('part', 'NMTOKEN', True)
    c_attributes['use'] = ('use', UseChoice_, True)
    c_attributes['encodingStyle'] = ('encoding_style', EncodingStyle_, False)
    c_attributes['namespace'] = ('namespace', 'anyURI', False)
    c_child_order.extend(['headerfault'])

    def __init__(self, headerfault=None, message=None, part=None, use=None, encoding_style=None, namespace=None, required=None, text=None, extension_elements=None, extension_attributes=None):
        wsdl.TExtensibilityElement_.__init__(self, required=required, text=text, extension_elements=extension_elements, extension_attributes=extension_attributes)
        self.headerfault = headerfault or []
        self.message = message
        self.part = part
        self.use = use
        self.encoding_style = encoding_style
        self.namespace = namespace