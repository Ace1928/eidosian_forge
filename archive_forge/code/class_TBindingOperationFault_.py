import saml2
from saml2 import SamlBase
class TBindingOperationFault_(TExtensibleDocumented_):
    """The http://schemas.xmlsoap.org/wsdl/:tBindingOperationFault element"""
    c_tag = 'tBindingOperationFault'
    c_namespace = NAMESPACE
    c_children = TExtensibleDocumented_.c_children.copy()
    c_attributes = TExtensibleDocumented_.c_attributes.copy()
    c_child_order = TExtensibleDocumented_.c_child_order[:]
    c_cardinality = TExtensibleDocumented_.c_cardinality.copy()
    c_attributes['name'] = ('name', 'NCName', True)

    def __init__(self, name=None, documentation=None, text=None, extension_elements=None, extension_attributes=None):
        TExtensibleDocumented_.__init__(self, documentation=documentation, text=text, extension_elements=extension_elements, extension_attributes=extension_attributes)
        self.name = name