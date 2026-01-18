import saml2
from saml2 import SamlBase
class RestrictedPasswordType_Length(RestrictedLengthType_):
    c_tag = 'Length'
    c_namespace = NAMESPACE
    c_children = RestrictedLengthType_.c_children.copy()
    c_attributes = RestrictedLengthType_.c_attributes.copy()
    c_child_order = RestrictedLengthType_.c_child_order[:]
    c_cardinality = RestrictedLengthType_.c_cardinality.copy()