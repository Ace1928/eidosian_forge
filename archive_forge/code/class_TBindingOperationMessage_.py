import saml2
from saml2 import SamlBase
class TBindingOperationMessage_(TExtensibleDocumented_):
    """The http://schemas.xmlsoap.org/wsdl/:tBindingOperationMessage element"""
    c_tag = 'tBindingOperationMessage'
    c_namespace = NAMESPACE
    c_children = TExtensibleDocumented_.c_children.copy()
    c_attributes = TExtensibleDocumented_.c_attributes.copy()
    c_child_order = TExtensibleDocumented_.c_child_order[:]
    c_cardinality = TExtensibleDocumented_.c_cardinality.copy()
    c_attributes['name'] = ('name', 'NCName', False)

    def __init__(self, name=None, documentation=None, text=None, extension_elements=None, extension_attributes=None):
        TExtensibleDocumented_.__init__(self, documentation=documentation, text=text, extension_elements=extension_elements, extension_attributes=extension_attributes)
        self.name = name