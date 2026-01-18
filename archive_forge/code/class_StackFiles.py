from openstack import resource
class StackFiles(resource.Resource):
    base_path = '/stacks/%(stack_name)s/%(stack_id)s/files'
    allow_create = False
    allow_list = False
    allow_fetch = True
    allow_delete = False
    allow_commit = False
    name = resource.URI('stack_name')
    stack_name = name
    id = resource.URI('stack_id')
    stack_id = id

    def fetch(self, session, base_path=None):
        request = self._prepare_request(requires_id=False, base_path=base_path)
        resp = session.get(request.url)
        return resp.json()