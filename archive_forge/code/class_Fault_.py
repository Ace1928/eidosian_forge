import saml2
from saml2 import SamlBase
class Fault_(SamlBase):
    """The http://schemas.xmlsoap.org/soap/envelope/:Fault element"""
    c_tag = 'Fault'
    c_namespace = NAMESPACE
    c_children = SamlBase.c_children.copy()
    c_attributes = SamlBase.c_attributes.copy()
    c_child_order = SamlBase.c_child_order[:]
    c_cardinality = SamlBase.c_cardinality.copy()
    c_children['{http://schemas.xmlsoap.org/soap/envelope/}faultcode'] = ('faultcode', Fault_faultcode)
    c_children['{http://schemas.xmlsoap.org/soap/envelope/}faultstring'] = ('faultstring', Fault_faultstring)
    c_children['{http://schemas.xmlsoap.org/soap/envelope/}faultactor'] = ('faultactor', Fault_faultactor)
    c_cardinality['faultactor'] = {'min': 0, 'max': 1}
    c_children['{http://schemas.xmlsoap.org/soap/envelope/}detail'] = ('detail', Fault_detail)
    c_cardinality['detail'] = {'min': 0, 'max': 1}
    c_child_order.extend(['faultcode', 'faultstring', 'faultactor', 'detail'])

    def __init__(self, faultcode=None, faultstring=None, faultactor=None, detail=None, text=None, extension_elements=None, extension_attributes=None):
        SamlBase.__init__(self, text=text, extension_elements=extension_elements, extension_attributes=extension_attributes)
        self.faultcode = faultcode
        self.faultstring = faultstring
        self.faultactor = faultactor
        self.detail = detail