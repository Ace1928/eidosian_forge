import saml2
from saml2 import SamlBase
class Envelope_(SamlBase):
    """The http://schemas.xmlsoap.org/soap/envelope/:Envelope element"""
    c_tag = 'Envelope'
    c_namespace = NAMESPACE
    c_children = SamlBase.c_children.copy()
    c_attributes = SamlBase.c_attributes.copy()
    c_child_order = SamlBase.c_child_order[:]
    c_cardinality = SamlBase.c_cardinality.copy()
    c_children['{http://schemas.xmlsoap.org/soap/envelope/}Header'] = ('header', Header_)
    c_cardinality['header'] = {'min': 0, 'max': 1}
    c_children['{http://schemas.xmlsoap.org/soap/envelope/}Body'] = ('body', Body_)
    c_child_order.extend(['header', 'body'])

    def __init__(self, header=None, body=None, text=None, extension_elements=None, extension_attributes=None):
        SamlBase.__init__(self, text=text, extension_elements=extension_elements, extension_attributes=extension_attributes)
        self.header = header
        self.body = body