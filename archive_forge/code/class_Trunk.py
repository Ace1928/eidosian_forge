from openstack.common import tag
from openstack import exceptions
from openstack import resource
from openstack import utils
class Trunk(resource.Resource, tag.TagMixin):
    resource_key = 'trunk'
    resources_key = 'trunks'
    base_path = '/trunks'
    _allow_unknown_attrs_in_body = True
    allow_create = True
    allow_fetch = True
    allow_commit = True
    allow_delete = True
    allow_list = True
    _query_mapping = resource.QueryParameters('name', 'description', 'port_id', 'status', 'sub_ports', 'project_id', is_admin_state_up='admin_state_up', **tag.TagMixin._tag_query_parameters)
    name = resource.Body('name')
    project_id = resource.Body('project_id', alias='tenant_id')
    tenant_id = resource.Body('tenant_id', deprecated=True)
    description = resource.Body('description')
    is_admin_state_up = resource.Body('admin_state_up', type=bool)
    port_id = resource.Body('port_id')
    status = resource.Body('status')
    sub_ports = resource.Body('sub_ports', type=list)

    def add_subports(self, session, subports):
        url = utils.urljoin('/trunks', self.id, 'add_subports')
        resp = session.put(url, json={'sub_ports': subports})
        exceptions.raise_from_response(resp)
        self._body.attributes.update(resp.json())
        return self

    def delete_subports(self, session, subports):
        url = utils.urljoin('/trunks', self.id, 'remove_subports')
        resp = session.put(url, json={'sub_ports': subports})
        exceptions.raise_from_response(resp)
        self._body.attributes.update(resp.json())
        return self

    def get_subports(self, session):
        url = utils.urljoin('/trunks', self.id, 'get_subports')
        resp = session.get(url)
        exceptions.raise_from_response(resp)
        self._body.attributes.update(resp.json())
        return resp.json()