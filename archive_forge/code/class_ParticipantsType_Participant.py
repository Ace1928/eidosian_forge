import saml2
from saml2 import SamlBase
from saml2.ws import wsaddr as wsa
from saml2.ws import wssec as wsse
from saml2.ws import wsutil as wsu
class ParticipantsType_Participant(ParticipantType_):
    c_tag = 'Participant'
    c_namespace = NAMESPACE
    c_children = ParticipantType_.c_children.copy()
    c_attributes = ParticipantType_.c_attributes.copy()
    c_child_order = ParticipantType_.c_child_order[:]
    c_cardinality = ParticipantType_.c_cardinality.copy()