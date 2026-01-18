from openstack import resource
class ShareSnapshotInstance(resource.Resource):
    resource_key = 'snapshot_instance'
    resources_key = 'snapshot_instances'
    base_path = '/snapshot-instances'
    allow_create = False
    allow_fetch = True
    allow_commit = False
    allow_delete = False
    allow_list = True
    allow_head = False
    created_at = resource.Body('created_at', type=str)
    progress = resource.Body('progress', type=str)
    provider_location = resource.Body('provider_location', type=str)
    share_id = resource.Body('share_id', type=str)
    share_instance_id = resource.Body('share_instance_id', type=str)
    snapshot_id = resource.Body('snapshot_id', type=str)
    status = resource.Body('status', type=str)
    updated_at = resource.Body('updated_at', type=str)