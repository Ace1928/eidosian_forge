from ansible_collections.openstack.cloud.plugins.module_utils.openstack import OpenStackModule
class IdentityProjectInfoModule(OpenStackModule):
    argument_spec = dict(domain=dict(), name=dict(), filters=dict(type='dict'))
    module_kwargs = dict(supports_check_mode=True)

    def run(self):
        filters = self.params['filters'] or {}
        domain_name_or_id = self.params['domain']
        if domain_name_or_id is not None:
            domain = self.conn.identity.find_domain(domain_name_or_id)
            if not domain:
                self.exit_json(changed=False, projects=[])
            filters['domain_id'] = domain.id
        projects = self.conn.search_projects(name_or_id=self.params['name'], filters=filters)
        self.exit_json(changed=False, projects=[p.to_dict(computed=False) for p in projects])