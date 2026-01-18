import saml2
from saml2 import SamlBase
class AuthenticatorBaseType_(SamlBase):
    """The urn:oasis:names:tc:SAML:2.0:ac:classes:InternetProtocolPassword:AuthenticatorBaseType element"""
    c_tag = 'AuthenticatorBaseType'
    c_namespace = NAMESPACE
    c_children = SamlBase.c_children.copy()
    c_attributes = SamlBase.c_attributes.copy()
    c_child_order = SamlBase.c_child_order[:]
    c_cardinality = SamlBase.c_cardinality.copy()
    c_children['{urn:oasis:names:tc:SAML:2.0:ac:classes:InternetProtocolPassword}Password'] = ('password', Password)
    c_children['{urn:oasis:names:tc:SAML:2.0:ac:classes:InternetProtocolPassword}IPAddress'] = ('ip_address', IPAddress)
    c_children['{urn:oasis:names:tc:SAML:2.0:ac:classes:InternetProtocolPassword}Extension'] = ('extension', [Extension])
    c_cardinality['extension'] = {'min': 0}
    c_child_order.extend(['password', 'ip_address', 'extension'])

    def __init__(self, password=None, ip_address=None, extension=None, text=None, extension_elements=None, extension_attributes=None):
        SamlBase.__init__(self, text=text, extension_elements=extension_elements, extension_attributes=extension_attributes)
        self.password = password
        self.ip_address = ip_address
        self.extension = extension or []