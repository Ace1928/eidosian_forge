import saml2
from saml2 import SamlBase
class TService_port(TPort_):
    c_tag = 'port'
    c_namespace = NAMESPACE
    c_children = TPort_.c_children.copy()
    c_attributes = TPort_.c_attributes.copy()
    c_child_order = TPort_.c_child_order[:]
    c_cardinality = TPort_.c_cardinality.copy()