from ansible_collections.openstack.cloud.plugins.module_utils.openstack import OpenStackModule
class IdentityGroupInfoModule(OpenStackModule):
    argument_spec = dict(domain=dict(), filters=dict(type='dict'), name=dict())
    module_kwargs = dict(supports_check_mode=True)

    def run(self):
        name = self.params['name']
        filters = self.params['filters'] or {}
        kwargs = {}
        domain_name_or_id = self.params['domain']
        if domain_name_or_id:
            domain = self.conn.identity.find_domain(domain_name_or_id)
            if domain is None:
                self.exit_json(changed=False, groups=[])
            kwargs['domain_id'] = domain['id']
        groups = self.conn.search_groups(name, filters, **kwargs)
        self.exit_json(changed=False, groups=[g.to_dict(computed=False) for g in groups])