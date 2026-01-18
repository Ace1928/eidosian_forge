import saml2
from saml2 import SamlBase
from saml2.schema import wsdl
class Headerfault(THeaderFault_):
    """The http://schemas.xmlsoap.org/wsdl/soap/:headerfault element"""
    c_tag = 'headerfault'
    c_namespace = NAMESPACE
    c_children = THeaderFault_.c_children.copy()
    c_attributes = THeaderFault_.c_attributes.copy()
    c_child_order = THeaderFault_.c_child_order[:]
    c_cardinality = THeaderFault_.c_cardinality.copy()