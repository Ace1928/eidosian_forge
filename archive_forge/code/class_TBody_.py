import saml2
from saml2 import SamlBase
from saml2.schema import wsdl
class TBody_(wsdl.TExtensibilityElement_):
    """The http://schemas.xmlsoap.org/wsdl/soap/:tBody element"""
    c_tag = 'tBody'
    c_namespace = NAMESPACE
    c_children = wsdl.TExtensibilityElement_.c_children.copy()
    c_attributes = wsdl.TExtensibilityElement_.c_attributes.copy()
    c_child_order = wsdl.TExtensibilityElement_.c_child_order[:]
    c_cardinality = wsdl.TExtensibilityElement_.c_cardinality.copy()
    c_attributes['parts'] = ('parts', 'NMTOKENS', False)
    c_attributes['encodingStyle'] = ('encoding_style', EncodingStyle_, False)
    c_attributes['use'] = ('use', UseChoice_, False)
    c_attributes['namespace'] = ('namespace', 'anyURI', False)

    def __init__(self, parts=None, encoding_style=None, use=None, namespace=None, required=None, text=None, extension_elements=None, extension_attributes=None):
        wsdl.TExtensibilityElement_.__init__(self, required=required, text=text, extension_elements=extension_elements, extension_attributes=extension_attributes)
        self.parts = parts
        self.encoding_style = encoding_style
        self.use = use
        self.namespace = namespace