import saml2
from saml2 import SamlBase
from saml2.schema import wsdl
class TFault_(TFaultRes_):
    """The http://schemas.xmlsoap.org/wsdl/soap/:tFault element"""
    c_tag = 'tFault'
    c_namespace = NAMESPACE
    c_children = TFaultRes_.c_children.copy()
    c_attributes = TFaultRes_.c_attributes.copy()
    c_child_order = TFaultRes_.c_child_order[:]
    c_cardinality = TFaultRes_.c_cardinality.copy()
    c_attributes['name'] = ('name', 'NCName', True)

    def __init__(self, name=None, required=None, parts=None, encoding_style=None, use=None, namespace=None, text=None, extension_elements=None, extension_attributes=None):
        TFaultRes_.__init__(self, required=required, parts=parts, encoding_style=encoding_style, use=use, namespace=namespace, text=text, extension_elements=extension_elements, extension_attributes=extension_attributes)
        self.name = name