import saml2
from saml2 import SamlBase
from saml2 import saml
from saml2 import xmldsig as ds
class IDPEntryType_(SamlBase):
    """The urn:oasis:names:tc:SAML:2.0:protocol:IDPEntryType element"""
    c_tag = 'IDPEntryType'
    c_namespace = NAMESPACE
    c_children = SamlBase.c_children.copy()
    c_attributes = SamlBase.c_attributes.copy()
    c_child_order = SamlBase.c_child_order[:]
    c_cardinality = SamlBase.c_cardinality.copy()
    c_attributes['ProviderID'] = ('provider_id', 'anyURI', True)
    c_attributes['Name'] = ('name', 'string', False)
    c_attributes['Loc'] = ('loc', 'anyURI', False)

    def __init__(self, provider_id=None, name=None, loc=None, text=None, extension_elements=None, extension_attributes=None):
        SamlBase.__init__(self, text=text, extension_elements=extension_elements, extension_attributes=extension_attributes)
        self.provider_id = provider_id
        self.name = name
        self.loc = loc