from openstack.common import tag
from openstack import resource
from openstack import utils
class QoSPolicy(resource.Resource, tag.TagMixin):
    resource_key = 'policy'
    resources_key = 'policies'
    base_path = '/qos/policies'
    _allow_unknown_attrs_in_body = True
    allow_create = True
    allow_fetch = True
    allow_commit = True
    allow_delete = True
    allow_list = True
    _query_mapping = resource.QueryParameters('name', 'description', 'is_default', 'project_id', 'sort_key', 'sort_dir', is_shared='shared', **tag.TagMixin._tag_query_parameters)
    name = resource.Body('name')
    project_id = resource.Body('project_id', alias='tenant_id')
    tenant_id = resource.Body('tenant_id', deprecated=True)
    description = resource.Body('description')
    is_default = resource.Body('is_default', type=bool)
    is_shared = resource.Body('shared', type=bool)
    rules = resource.Body('rules')

    def set_tags(self, session, tags):
        url = utils.urljoin('/policies', self.id, 'tags')
        session.put(url, json={'tags': tags})
        self._body.attributes.update({'tags': tags})
        return self