from openstack import resource
class VolumeAttachment(resource.Resource):
    resource_key = 'volumeAttachment'
    resources_key = 'volumeAttachments'
    base_path = '/servers/%(server_id)s/os-volume_attachments'
    allow_create = True
    allow_fetch = True
    allow_commit = True
    allow_delete = True
    allow_list = True
    _query_mapping = resource.QueryParameters('limit', 'offset')
    server_id = resource.URI('server_id')
    device = resource.Body('device')
    id = resource.Body('id')
    volume_id = resource.Body('volumeId', alternate_id=True)
    attachment_id = resource.Body('attachment_id')
    bdm_id = resource.Body('bdm_uuid')
    tag = resource.Body('tag')
    delete_on_termination = resource.Body('delete_on_termination')
    _max_microversion = '2.89'