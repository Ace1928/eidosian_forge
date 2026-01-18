import saml2
from saml2 import SamlBase
from saml2.ws import wsaddr as wsa
from saml2.ws import wssec as wsse
from saml2.ws import wsutil as wsu
class BinarySecret(BinarySecretType_):
    """The http://docs.oasis-open.org/ws-sx/ws-trust/200512/:BinarySecret element"""
    c_tag = 'BinarySecret'
    c_namespace = NAMESPACE
    c_children = BinarySecretType_.c_children.copy()
    c_attributes = BinarySecretType_.c_attributes.copy()
    c_child_order = BinarySecretType_.c_child_order[:]
    c_cardinality = BinarySecretType_.c_cardinality.copy()