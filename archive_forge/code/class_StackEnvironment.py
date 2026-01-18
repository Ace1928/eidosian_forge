from openstack import resource
class StackEnvironment(resource.Resource):
    base_path = '/stacks/%(stack_name)s/%(stack_id)s/environment'
    allow_create = False
    allow_list = False
    allow_fetch = True
    allow_delete = False
    allow_commit = False
    name = resource.URI('stack_name')
    stack_name = name
    id = resource.URI('stack_id')
    stack_id = id
    encrypted_param_names = resource.Body('encrypted_param_names')
    event_sinks = resource.Body('event_sinks')
    parameter_defaults = resource.Body('parameter_defaults')
    parameters = resource.Body('parameters', type=dict)
    resource_registry = resource.Body('resource_registry', type=dict)