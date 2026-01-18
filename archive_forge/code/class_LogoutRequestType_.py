import saml2
from saml2 import SamlBase
from saml2 import saml
from saml2 import xmldsig as ds
class LogoutRequestType_(RequestAbstractType_):
    """The urn:oasis:names:tc:SAML:2.0:protocol:LogoutRequestType element"""
    c_tag = 'LogoutRequestType'
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
    c_children['{urn:oasis:names:tc:SAML:2.0:protocol}SessionIndex'] = ('session_index', [SessionIndex])
    c_cardinality['session_index'] = {'min': 0}
    c_attributes['Reason'] = ('reason', 'string', False)
    c_attributes['NotOnOrAfter'] = ('not_on_or_after', 'dateTime', False)
    c_child_order.extend(['base_id', 'name_id', 'encrypted_id', 'session_index'])

    def __init__(self, base_id=None, name_id=None, encrypted_id=None, session_index=None, reason=None, not_on_or_after=None, issuer=None, signature=None, extensions=None, id=None, version=None, issue_instant=None, destination=None, consent=None, text=None, extension_elements=None, extension_attributes=None):
        RequestAbstractType_.__init__(self, issuer=issuer, signature=signature, extensions=extensions, id=id, version=version, issue_instant=issue_instant, destination=destination, consent=consent, text=text, extension_elements=extension_elements, extension_attributes=extension_attributes)
        self.base_id = base_id
        self.name_id = name_id
        self.encrypted_id = encrypted_id
        self.session_index = session_index or []
        self.reason = reason
        self.not_on_or_after = not_on_or_after