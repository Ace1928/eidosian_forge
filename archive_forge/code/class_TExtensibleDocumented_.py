import saml2
from saml2 import SamlBase
class TExtensibleDocumented_(TDocumented_):
    """The http://schemas.xmlsoap.org/wsdl/:tExtensibleDocumented element"""
    c_tag = 'tExtensibleDocumented'
    c_namespace = NAMESPACE
    c_children = TDocumented_.c_children.copy()
    c_attributes = TDocumented_.c_attributes.copy()
    c_child_order = TDocumented_.c_child_order[:]
    c_cardinality = TDocumented_.c_cardinality.copy()