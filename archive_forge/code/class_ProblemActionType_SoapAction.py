import saml2
from saml2 import SamlBase
class ProblemActionType_SoapAction(SamlBase):
    c_tag = 'SoapAction'
    c_namespace = NAMESPACE
    c_value_type = {'base': 'anyURI'}
    c_children = SamlBase.c_children.copy()
    c_attributes = SamlBase.c_attributes.copy()
    c_child_order = SamlBase.c_child_order[:]
    c_cardinality = SamlBase.c_cardinality.copy()