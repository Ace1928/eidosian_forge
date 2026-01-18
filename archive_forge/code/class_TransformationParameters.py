import saml2
from saml2 import SamlBase
class TransformationParameters(TransformationParametersType_):
    """The http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-wssecurity-secext-1.0.xsd:TransformationParameters element"""
    c_tag = 'TransformationParameters'
    c_namespace = NAMESPACE
    c_children = TransformationParametersType_.c_children.copy()
    c_attributes = TransformationParametersType_.c_attributes.copy()
    c_child_order = TransformationParametersType_.c_child_order[:]
    c_cardinality = TransformationParametersType_.c_cardinality.copy()