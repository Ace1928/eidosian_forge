from openstack.common import tag
from openstack.network.v2 import _base
from openstack import resource
class FloatingIP(_base.NetworkResource, tag.TagMixin):
    name_attribute = 'floating_ip_address'
    resource_name = 'floating ip'
    resource_key = 'floatingip'
    resources_key = 'floatingips'
    base_path = '/floatingips'
    allow_create = True
    allow_fetch = True
    allow_commit = True
    allow_delete = True
    allow_list = True
    _query_mapping = resource.QueryParameters('description', 'fixed_ip_address', 'floating_ip_address', 'floating_network_id', 'port_id', 'router_id', 'status', 'subnet_id', 'project_id', 'tenant_id', 'sort_key', 'sort_dir', tenant_id='project_id', **tag.TagMixin._tag_query_parameters)
    created_at = resource.Body('created_at')
    description = resource.Body('description')
    dns_domain = resource.Body('dns_domain')
    dns_name = resource.Body('dns_name')
    fixed_ip_address = resource.Body('fixed_ip_address')
    floating_ip_address = resource.Body('floating_ip_address')
    name = floating_ip_address
    floating_network_id = resource.Body('floating_network_id')
    port_details = resource.Body('port_details', type=dict)
    port_id = resource.Body('port_id')
    qos_policy_id = resource.Body('qos_policy_id')
    project_id = resource.Body('project_id', alias='tenant_id')
    tenant_id = resource.Body('tenant_id', deprecated=True)
    router_id = resource.Body('router_id')
    status = resource.Body('status')
    updated_at = resource.Body('updated_at')
    subnet_id = resource.Body('subnet_id')

    @classmethod
    def find_available(cls, session):
        for ip in cls.list(session):
            if not ip.port_id:
                return ip
        return None