import saml2
from saml2 import SamlBase
class ProblemActionType_(SamlBase):
    """The http://www.w3.org/2005/08/addressing:ProblemActionType element"""
    c_tag = 'ProblemActionType'
    c_namespace = NAMESPACE
    c_children = SamlBase.c_children.copy()
    c_attributes = SamlBase.c_attributes.copy()
    c_child_order = SamlBase.c_child_order[:]
    c_cardinality = SamlBase.c_cardinality.copy()
    c_children['{http://www.w3.org/2005/08/addressing}Action'] = ('action', Action)
    c_cardinality['action'] = {'min': 0, 'max': 1}
    c_children['{http://www.w3.org/2005/08/addressing}SoapAction'] = ('soap_action', ProblemActionType_SoapAction)
    c_cardinality['soap_action'] = {'min': 0, 'max': 1}
    c_child_order.extend(['action', 'soap_action'])

    def __init__(self, action=None, soap_action=None, text=None, extension_elements=None, extension_attributes=None):
        SamlBase.__init__(self, text=text, extension_elements=extension_elements, extension_attributes=extension_attributes)
        self.action = action
        self.soap_action = soap_action