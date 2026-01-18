import saml2
from saml2 import SamlBase
class Fault_detail(Detail_):
    c_tag = 'detail'
    c_namespace = NAMESPACE
    c_children = Detail_.c_children.copy()
    c_attributes = Detail_.c_attributes.copy()
    c_child_order = Detail_.c_child_order[:]
    c_cardinality = Detail_.c_cardinality.copy()