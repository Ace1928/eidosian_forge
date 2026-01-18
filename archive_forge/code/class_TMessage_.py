import saml2
from saml2 import SamlBase
class TMessage_(TExtensibleDocumented_):
    """The http://schemas.xmlsoap.org/wsdl/:tMessage element"""
    c_tag = 'tMessage'
    c_namespace = NAMESPACE
    c_children = TExtensibleDocumented_.c_children.copy()
    c_attributes = TExtensibleDocumented_.c_attributes.copy()
    c_child_order = TExtensibleDocumented_.c_child_order[:]
    c_cardinality = TExtensibleDocumented_.c_cardinality.copy()
    c_children['{http://schemas.xmlsoap.org/wsdl/}part'] = ('part', [TMessage_part])
    c_cardinality['part'] = {'min': 0}
    c_attributes['name'] = ('name', 'NCName', True)
    c_child_order.extend(['part'])

    def __init__(self, part=None, name=None, documentation=None, text=None, extension_elements=None, extension_attributes=None):
        TExtensibleDocumented_.__init__(self, documentation=documentation, text=text, extension_elements=extension_elements, extension_attributes=extension_attributes)
        self.part = part or []
        self.name = name