import saml2
from saml2 import SamlBase
from saml2 import saml
from saml2 import xmldsig as ds
class NameIDMappingResponseType_(StatusResponseType_):
    """
    The urn:oasis:names:tc:SAML:2.0:protocol:NameIDMappingResponseType element
    """
    c_tag = 'NameIDMappingResponseType'
    c_namespace = NAMESPACE
    c_children = StatusResponseType_.c_children.copy()
    c_attributes = StatusResponseType_.c_attributes.copy()
    c_child_order = StatusResponseType_.c_child_order[:]
    c_cardinality = StatusResponseType_.c_cardinality.copy()
    c_children['{urn:oasis:names:tc:SAML:2.0:assertion}NameID'] = ('name_id', saml.NameID)
    c_cardinality['name_id'] = {'min': 0, 'max': 1}
    c_children['{urn:oasis:names:tc:SAML:2.0:assertion}EncryptedID'] = ('encrypted_id', saml.EncryptedID)
    c_cardinality['encrypted_id'] = {'min': 0, 'max': 1}
    c_child_order.extend(['name_id', 'encrypted_id'])

    def __init__(self, name_id=None, encrypted_id=None, issuer=None, signature=None, extensions=None, status=None, id=None, in_response_to=None, version=None, issue_instant=None, destination=None, consent=None, text=None, extension_elements=None, extension_attributes=None):
        StatusResponseType_.__init__(self, issuer=issuer, signature=signature, extensions=extensions, status=status, id=id, in_response_to=in_response_to, version=version, issue_instant=issue_instant, destination=destination, consent=consent, text=text, extension_elements=extension_elements, extension_attributes=extension_attributes)
        self.name_id = name_id
        self.encrypted_id = encrypted_id