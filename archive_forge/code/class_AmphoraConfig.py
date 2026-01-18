from openstack import resource
class AmphoraConfig(resource.Resource):
    base_path = '/octavia/amphorae/%(amphora_id)s/config'
    allow_create = False
    allow_fetch = False
    allow_commit = True
    allow_delete = False
    allow_list = False
    allow_empty_commit = True
    requires_id = False
    amphora_id = resource.URI('amphora_id')

    def commit(self, session, base_path=None):
        return super(AmphoraConfig, self).commit(session, base_path=base_path, has_body=False)