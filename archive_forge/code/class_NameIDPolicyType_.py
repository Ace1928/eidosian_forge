import saml2
from saml2 import SamlBase
from saml2 import saml
from saml2 import xmldsig as ds
class NameIDPolicyType_(SamlBase):
    """The urn:oasis:names:tc:SAML:2.0:protocol:NameIDPolicyType element"""
    c_tag = 'NameIDPolicyType'
    c_namespace = NAMESPACE
    c_children = SamlBase.c_children.copy()
    c_attributes = SamlBase.c_attributes.copy()
    c_child_order = SamlBase.c_child_order[:]
    c_cardinality = SamlBase.c_cardinality.copy()
    c_attributes['Format'] = ('format', 'anyURI', False)
    c_attributes['SPNameQualifier'] = ('sp_name_qualifier', 'string', False)
    c_attributes['AllowCreate'] = ('allow_create', 'boolean', False)

    def __init__(self, format=None, sp_name_qualifier=None, allow_create=None, text=None, extension_elements=None, extension_attributes=None):
        SamlBase.__init__(self, text=text, extension_elements=extension_elements, extension_attributes=extension_attributes)
        self.format = format
        self.sp_name_qualifier = sp_name_qualifier
        self.allow_create = allow_create