from openstack import resource
class UserMessage(resource.Resource):
    resource_key = 'message'
    resources_key = 'messages'
    base_path = '/messages'
    allow_fetch = True
    allow_commit = False
    allow_delete = True
    allow_list = True
    allow_head = False
    _query_mapping = resource.QueryParameters('message_id')
    _max_microversion = '2.37'
    action_id = resource.Body('action_id', type=str)
    created_at = resource.Body('created_at', type=str)
    detail_id = resource.Body('detail_id', type=str)
    expires_at = resource.Body('expires_at', type=str)
    message_level = resource.Body('message_level', type=str)
    project_id = resource.Body('project_id', type=str)
    request_id = resource.Body('request_id', type=str)
    resource_id = resource.Body('resource_id', type=str)
    resource_type = resource.Body('resource_type', type=str)
    user_message = resource.Body('user_message', type=str)