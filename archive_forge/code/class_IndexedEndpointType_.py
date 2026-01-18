import saml2
from saml2 import SamlBase
from saml2 import saml
from saml2 import xmldsig as ds
from saml2 import xmlenc as xenc
class IndexedEndpointType_(EndpointType_):
    """The urn:oasis:names:tc:SAML:2.0:metadata:IndexedEndpointType element"""
    c_tag = 'IndexedEndpointType'
    c_namespace = NAMESPACE
    c_children = EndpointType_.c_children.copy()
    c_attributes = EndpointType_.c_attributes.copy()
    c_child_order = EndpointType_.c_child_order[:]
    c_cardinality = EndpointType_.c_cardinality.copy()
    c_attributes['index'] = ('index', 'unsignedShort', True)
    c_attributes['isDefault'] = ('is_default', 'boolean', False)

    def __init__(self, index=None, is_default=None, binding=None, location=None, response_location=None, text=None, extension_elements=None, extension_attributes=None):
        EndpointType_.__init__(self, binding=binding, location=location, response_location=response_location, text=text, extension_elements=extension_elements, extension_attributes=extension_attributes)
        self.index = index
        self.is_default = is_default