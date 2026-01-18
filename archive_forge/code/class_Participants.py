import saml2
from saml2 import SamlBase
from saml2.ws import wsaddr as wsa
from saml2.ws import wssec as wsse
from saml2.ws import wsutil as wsu
class Participants(ParticipantsType_):
    """The http://docs.oasis-open.org/ws-sx/ws-trust/200512/:Participants element"""
    c_tag = 'Participants'
    c_namespace = NAMESPACE
    c_children = ParticipantsType_.c_children.copy()
    c_attributes = ParticipantsType_.c_attributes.copy()
    c_child_order = ParticipantsType_.c_child_order[:]
    c_cardinality = ParticipantsType_.c_cardinality.copy()