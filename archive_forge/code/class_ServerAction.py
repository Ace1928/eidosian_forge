from openstack import resource
class ServerAction(resource.Resource):
    resource_key = 'instanceAction'
    resources_key = 'instanceActions'
    base_path = '/servers/%(server_id)s/os-instance-actions'
    allow_fetch = True
    allow_list = True
    server_id = resource.URI('server_id')
    action = resource.Body('action')
    request_id = resource.Body('request_id', alternate_id=True)
    user_id = resource.Body('user_id')
    project_id = resource.Body('project_id')
    start_time = resource.Body('start_time')
    message = resource.Body('message')
    events = resource.Body('events', type=list, list_type=ServerActionEvent)
    _max_microversion = '2.84'
    _query_mapping = resource.QueryParameters(changes_since='changes-since', changes_before='changes-before')