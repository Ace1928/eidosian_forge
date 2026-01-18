from openstack import resource
class SoftwareDeployment(resource.Resource):
    resource_key = 'software_deployment'
    resources_key = 'software_deployments'
    base_path = '/software_deployments'
    allow_create = True
    allow_list = True
    allow_fetch = True
    allow_delete = True
    allow_commit = True
    action = resource.Body('action')
    config_id = resource.Body('config_id')
    input_values = resource.Body('input_values', type=dict)
    output_values = resource.Body('output_values', type=dict)
    server_id = resource.Body('server_id')
    stack_user_project_id = resource.Body('stack_user_project_id')
    status = resource.Body('status')
    status_reason = resource.Body('status_reason')
    created_at = resource.Body('creation_time')
    updated_at = resource.Body('updated_time')

    def create(self, session, base_path=None):
        return super(SoftwareDeployment, self).create(session, prepend_key=False, base_path=base_path)

    def commit(self, session, base_path=None):
        return super(SoftwareDeployment, self).commit(session, prepend_key=False, base_path=base_path)