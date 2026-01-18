import saml2
from saml2 import SamlBase
from saml2 import saml
from saml2 import xmldsig as ds
class StatusResponseType_(SamlBase):
    """The urn:oasis:names:tc:SAML:2.0:protocol:StatusResponseType element"""
    c_tag = 'StatusResponseType'
    c_namespace = NAMESPACE
    c_children = SamlBase.c_children.copy()
    c_attributes = SamlBase.c_attributes.copy()
    c_child_order = SamlBase.c_child_order[:]
    c_cardinality = SamlBase.c_cardinality.copy()
    c_children['{urn:oasis:names:tc:SAML:2.0:assertion}Issuer'] = ('issuer', saml.Issuer)
    c_cardinality['issuer'] = {'min': 0, 'max': 1}
    c_children['{http://www.w3.org/2000/09/xmldsig#}Signature'] = ('signature', ds.Signature)
    c_cardinality['signature'] = {'min': 0, 'max': 1}
    c_children['{urn:oasis:names:tc:SAML:2.0:protocol}Extensions'] = ('extensions', Extensions)
    c_cardinality['extensions'] = {'min': 0, 'max': 1}
    c_children['{urn:oasis:names:tc:SAML:2.0:protocol}Status'] = ('status', Status)
    c_attributes['ID'] = ('id', 'ID', True)
    c_attributes['InResponseTo'] = ('in_response_to', 'NCName', False)
    c_attributes['Version'] = ('version', 'string', True)
    c_attributes['IssueInstant'] = ('issue_instant', 'dateTime', True)
    c_attributes['Destination'] = ('destination', 'anyURI', False)
    c_attributes['Consent'] = ('consent', 'anyURI', False)
    c_child_order.extend(['issuer', 'signature', 'extensions', 'status'])

    def __init__(self, issuer=None, signature=None, extensions=None, status=None, id=None, in_response_to=None, version=None, issue_instant=None, destination=None, consent=None, text=None, extension_elements=None, extension_attributes=None):
        SamlBase.__init__(self, text=text, extension_elements=extension_elements, extension_attributes=extension_attributes)
        self.issuer = issuer
        self.signature = signature
        self.extensions = extensions
        self.status = status
        self.id = id
        self.in_response_to = in_response_to
        self.version = version
        self.issue_instant = issue_instant
        self.destination = destination
        self.consent = consent