import saml2
from saml2 import SamlBase
class Q(CryptoBinary_):
    c_tag = 'Q'
    c_namespace = NAMESPACE
    c_children = CryptoBinary_.c_children.copy()
    c_attributes = CryptoBinary_.c_attributes.copy()
    c_child_order = CryptoBinary_.c_child_order[:]
    c_cardinality = CryptoBinary_.c_cardinality.copy()