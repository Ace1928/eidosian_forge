import saml2
from saml2 import SamlBase
from saml2.ws import wsaddr as wsa
from saml2.ws import wssec as wsse
from saml2.ws import wsutil as wsu
class ComputedKeyEnum_(SamlBase):
    """The http://docs.oasis-open.org/ws-sx/ws-trust/200512/:ComputedKeyEnum element"""
    c_tag = 'ComputedKeyEnum'
    c_namespace = NAMESPACE
    c_value_type = {'base': 'xs:anyURI', 'enumeration': ['http://docs.oasis-open.org/ws-sx/ws-trust/200512/CK/PSHA1', 'http://docs.oasis-open.org/ws-sx/ws-trust/200512/CK/HASH']}
    c_children = SamlBase.c_children.copy()
    c_attributes = SamlBase.c_attributes.copy()
    c_child_order = SamlBase.c_child_order[:]
    c_cardinality = SamlBase.c_cardinality.copy()