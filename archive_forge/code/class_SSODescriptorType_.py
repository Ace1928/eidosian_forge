import saml2
from saml2 import SamlBase
from saml2 import saml
from saml2 import xmldsig as ds
from saml2 import xmlenc as xenc
class SSODescriptorType_(RoleDescriptorType_):
    """The urn:oasis:names:tc:SAML:2.0:metadata:SSODescriptorType element"""
    c_tag = 'SSODescriptorType'
    c_namespace = NAMESPACE
    c_children = RoleDescriptorType_.c_children.copy()
    c_attributes = RoleDescriptorType_.c_attributes.copy()
    c_child_order = RoleDescriptorType_.c_child_order[:]
    c_cardinality = RoleDescriptorType_.c_cardinality.copy()
    c_children['{urn:oasis:names:tc:SAML:2.0:metadata}ArtifactResolutionService'] = ('artifact_resolution_service', [ArtifactResolutionService])
    c_cardinality['artifact_resolution_service'] = {'min': 0}
    c_children['{urn:oasis:names:tc:SAML:2.0:metadata}SingleLogoutService'] = ('single_logout_service', [SingleLogoutService])
    c_cardinality['single_logout_service'] = {'min': 0}
    c_children['{urn:oasis:names:tc:SAML:2.0:metadata}ManageNameIDService'] = ('manage_name_id_service', [ManageNameIDService])
    c_cardinality['manage_name_id_service'] = {'min': 0}
    c_children['{urn:oasis:names:tc:SAML:2.0:metadata}NameIDFormat'] = ('name_id_format', [NameIDFormat])
    c_cardinality['name_id_format'] = {'min': 0}
    c_child_order.extend(['artifact_resolution_service', 'single_logout_service', 'manage_name_id_service', 'name_id_format'])

    def __init__(self, artifact_resolution_service=None, single_logout_service=None, manage_name_id_service=None, name_id_format=None, signature=None, extensions=None, key_descriptor=None, organization=None, contact_person=None, id=None, valid_until=None, cache_duration=None, protocol_support_enumeration=None, error_url=None, text=None, extension_elements=None, extension_attributes=None):
        RoleDescriptorType_.__init__(self, signature=signature, extensions=extensions, key_descriptor=key_descriptor, organization=organization, contact_person=contact_person, id=id, valid_until=valid_until, cache_duration=cache_duration, protocol_support_enumeration=protocol_support_enumeration, error_url=error_url, text=text, extension_elements=extension_elements, extension_attributes=extension_attributes)
        self.artifact_resolution_service = artifact_resolution_service or []
        self.single_logout_service = single_logout_service or []
        self.manage_name_id_service = manage_name_id_service or []
        self.name_id_format = name_id_format or []