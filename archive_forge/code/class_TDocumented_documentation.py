import saml2
from saml2 import SamlBase
class TDocumented_documentation(TDocumentation_):
    c_tag = 'documentation'
    c_namespace = NAMESPACE
    c_children = TDocumentation_.c_children.copy()
    c_attributes = TDocumentation_.c_attributes.copy()
    c_child_order = TDocumentation_.c_child_order[:]
    c_cardinality = TDocumentation_.c_cardinality.copy()