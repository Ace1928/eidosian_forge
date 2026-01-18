from openstack import resource
class VMove(resource.Resource):
    resource_key = 'vmove'
    resources_key = 'vmoves'
    base_path = '/notifications/%(notification_id)s/vmoves'
    allow_list = True
    allow_fetch = True
    _query_mapping = resource.QueryParameters('sort_key', 'sort_dir', 'type', 'status')
    id = resource.Body('id')
    uuid = resource.Body('uuid')
    notification_id = resource.URI('notification_id')
    created_at = resource.Body('created_at')
    updated_at = resource.Body('updated_at')
    server_id = resource.Body('instance_uuid')
    server_name = resource.Body('instance_name')
    source_host = resource.Body('source_host')
    dest_host = resource.Body('dest_host')
    start_time = resource.Body('start_time')
    end_time = resource.Body('end_time')
    status = resource.Body('status')
    type = resource.Body('type')
    message = resource.Body('message')