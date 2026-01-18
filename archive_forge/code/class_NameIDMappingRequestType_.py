import saml2
from saml2 import SamlBase
from saml2 import saml
from saml2 import xmldsig as ds
class NameIDMappingRequestType_(RequestAbstractType_):
    """The urn:oasis:names:tc:SAML:2.0:protocol:NameIDMappingRequestType
    element"""
    c_tag = 'NameIDMappingRequestType'
    c_namespace = NAMESPACE
    c_children = RequestAbstractType_.c_children.copy()
    c_attributes = RequestAbstractType_.c_attributes.copy()
    c_child_order = RequestAbstractType_.c_child_order[:]
    c_cardinality = RequestAbstractType_.c_cardinality.copy()
    c_children['{urn:oasis:names:tc:SAML:2.0:assertion}BaseID'] = ('base_id', saml.BaseID)
    c_cardinality['base_id'] = {'min': 0, 'max': 1}
    c_children['{urn:oasis:names:tc:SAML:2.0:assertion}NameID'] = ('name_id', saml.NameID)
    c_cardinality['name_id'] = {'min': 0, 'max': 1}
    c_children['{urn:oasis:names:tc:SAML:2.0:assertion}EncryptedID'] = ('encrypted_id', saml.EncryptedID)
    c_cardinality['encrypted_id'] = {'min': 0, 'max': 1}
    c_children['{urn:oasis:names:tc:SAML:2.0:protocol}NameIDPolicy'] = ('name_id_policy', NameIDPolicy)
    c_child_order.extend(['base_id', 'name_id', 'encrypted_id', 'name_id_policy'])

    def __init__(self, base_id=None, name_id=None, encrypted_id=None, name_id_policy=None, issuer=None, signature=None, extensions=None, id=None, version=None, issue_instant=None, destination=None, consent=None, text=None, extension_elements=None, extension_attributes=None):
        RequestAbstractType_.__init__(self, issuer=issuer, signature=signature, extensions=extensions, id=id, version=version, issue_instant=issue_instant, destination=destination, consent=consent, text=text, extension_elements=extension_elements, extension_attributes=extension_attributes)
        self.base_id = base_id
        self.name_id = name_id
        self.encrypted_id = encrypted_id
        self.name_id_policy = name_id_policy