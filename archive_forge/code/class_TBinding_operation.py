import saml2
from saml2 import SamlBase
class TBinding_operation(TBindingOperation_):
    c_tag = 'operation'
    c_namespace = NAMESPACE
    c_children = TBindingOperation_.c_children.copy()
    c_attributes = TBindingOperation_.c_attributes.copy()
    c_child_order = TBindingOperation_.c_child_order[:]
    c_cardinality = TBindingOperation_.c_cardinality.copy()