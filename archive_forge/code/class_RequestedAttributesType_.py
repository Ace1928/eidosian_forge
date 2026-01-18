import saml2
from saml2 import SamlBase
from saml2 import saml
class RequestedAttributesType_(SamlBase):
    """The http://eidas.europa.eu/saml-extensions:RequestedAttributesType element"""
    c_tag = 'RequestedAttributesType'
    c_namespace = NAMESPACE
    c_children = SamlBase.c_children.copy()
    c_attributes = SamlBase.c_attributes.copy()
    c_child_order = SamlBase.c_child_order[:]
    c_cardinality = SamlBase.c_cardinality.copy()
    c_children['{http://eidas.europa.eu/saml-extensions}RequestedAttribute'] = ('requested_attribute', [RequestedAttribute])
    c_cardinality['requested_attribute'] = {'min': 0}
    c_child_order.extend(['requested_attribute'])

    def __init__(self, requested_attribute=None, text=None, extension_elements=None, extension_attributes=None):
        SamlBase.__init__(self, text=text, extension_elements=extension_elements, extension_attributes=extension_attributes)
        self.requested_attribute = requested_attribute or []