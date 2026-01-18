from ansible_collections.openstack.cloud.plugins.module_utils.openstack import OpenStackModule
class IdentityRoleInfoModule(OpenStackModule):
    argument_spec = dict(domain_id=dict(), name=dict())
    module_kwargs = dict(supports_check_mode=True)

    def run(self):
        kwargs = dict(((k, self.params[k]) for k in ['domain_id'] if self.params[k] is not None))
        name_or_id = self.params['name']
        if name_or_id is not None:
            kwargs['name_or_id'] = name_or_id
        self.exit_json(changed=False, roles=[r.to_dict(computed=False) for r in self.conn.search_roles(**kwargs)])