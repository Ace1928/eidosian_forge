from openstack import exceptions
from openstack import resource
from openstack import utils
class TypeEncryption(resource.Resource):
    resource_key = 'encryption'
    resources_key = 'encryption'
    base_path = '/types/%(volume_type_id)s/encryption'
    allow_fetch = True
    allow_create = True
    allow_delete = True
    allow_list = False
    allow_commit = True
    cipher = resource.Body('cipher')
    control_location = resource.Body('control_location')
    created_at = resource.Body('created_at')
    deleted = resource.Body('deleted')
    deleted_at = resource.Body('deleted_at')
    encryption_id = resource.Body('encryption_id', alternate_id=True)
    key_size = resource.Body('key_size')
    provider = resource.Body('provider')
    updated_at = resource.Body('updated_at')
    volume_type_id = resource.URI('volume_type_id')