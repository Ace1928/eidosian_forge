import saml2
from saml2 import SamlBase
from saml2 import xmldsig as ds
class ReferenceList(SamlBase):
    """The http://www.w3.org/2001/04/xmlenc#:ReferenceList element"""
    c_tag = 'ReferenceList'
    c_namespace = NAMESPACE
    c_children = SamlBase.c_children.copy()
    c_attributes = SamlBase.c_attributes.copy()
    c_child_order = SamlBase.c_child_order[:]
    c_cardinality = SamlBase.c_cardinality.copy()
    c_children['{http://www.w3.org/2001/04/xmlenc#}DataReference'] = ('data_reference', [DataReference])
    c_cardinality['data_reference'] = {'min': 0}
    c_children['{http://www.w3.org/2001/04/xmlenc#}KeyReference'] = ('key_reference', [KeyReference])
    c_cardinality['key_reference'] = {'min': 0}
    c_child_order.extend(['data_reference', 'key_reference'])

    def __init__(self, data_reference=None, key_reference=None, text=None, extension_elements=None, extension_attributes=None):
        SamlBase.__init__(self, text=text, extension_elements=extension_elements, extension_attributes=extension_attributes)
        self.data_reference = data_reference or []
        self.key_reference = key_reference or []