import saml2
from saml2 import SamlBase
from saml2 import md
class RegistrationInfoType_(SamlBase):
    """The urn:oasis:names:tc:SAML:2.0:metadata:dri:RegistrationInfoType
    element"""
    c_tag = 'RegistrationInfoType'
    c_namespace = NAMESPACE
    c_children = SamlBase.c_children.copy()
    c_attributes = SamlBase.c_attributes.copy()
    c_child_order = SamlBase.c_child_order[:]
    c_cardinality = SamlBase.c_cardinality.copy()
    c_children['{urn:oasis:names:tc:SAML:2.0:metadata:dri}RegistrationAuthority'] = ('registration_authority', RegistrationAuthority)
    c_children['{urn:oasis:names:tc:SAML:2.0:metadata:dri}RegistrationInstant'] = ('registration_instant', RegistrationInstant)
    c_children['{urn:oasis:names:tc:SAML:2.0:metadata:dri}RegistrationPolicy'] = ('registration_policy', RegistrationPolicy)
    c_cardinality['registration_policy'] = {'min': 0, 'max': 1}
    c_child_order.extend(['registration_authority', 'registration_instant', 'registration_policy'])

    def __init__(self, registration_authority=None, registration_instant=None, registration_policy=None, text=None, extension_elements=None, extension_attributes=None):
        SamlBase.__init__(self, text=text, extension_elements=extension_elements, extension_attributes=extension_attributes)
        self.registration_authority = registration_authority
        self.registration_instant = registration_instant
        self.registration_policy = registration_policy