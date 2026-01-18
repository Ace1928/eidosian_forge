from ansible_collections.openstack.cloud.plugins.module_utils.openstack import OpenStackModule
class ComputeFlavorAccess(OpenStackModule):
    argument_spec = dict(name=dict(required=True), project=dict(required=True), project_domain=dict(), state=dict(default='present', choices=['absent', 'present']))
    module_kwargs = dict(supports_check_mode=True)

    def _project_and_project_domain(self):
        project_name_or_id = self.params['project']
        project_domain_name_or_id = self.params['project_domain']
        if project_domain_name_or_id:
            domain_id = self.conn.identity.find_domain(project_domain_name_or_id, ignore_missing=False).id
        else:
            domain_id = None
        kwargs = dict() if domain_id is None else dict(domain_id=domain_id)
        if project_name_or_id:
            project_id = self.conn.identity.find_project(project_name_or_id, *kwargs, ignore_missing=False).id
        else:
            project_id = None
        return (project_id, domain_id)

    def run(self):
        name_or_id = self.params['name']
        flavor = self.conn.compute.find_flavor(name_or_id, ignore_missing=False)
        state = self.params['state']
        if state == 'present' and flavor.is_public:
            raise ValueError('access can only be granted to private flavors')
        project_id, domain_id = self._project_and_project_domain()
        flavor_access = self.conn.compute.get_flavor_access(flavor.id)
        project_ids = [access.get('tenant_id') for access in flavor_access]
        if project_id in project_ids and state == 'present' or (project_id not in project_ids and state == 'absent'):
            self.exit_json(changed=False, flavor=flavor.to_dict(computed=False))
        if self.ansible.check_mode:
            self.exit_json(changed=True, flavor=flavor.to_dict(computed=False))
        if project_id in project_ids:
            self.conn.compute.flavor_remove_tenant_access(flavor.id, project_id)
        else:
            self.conn.compute.flavor_add_tenant_access(flavor.id, project_id)
        self.exit_json(changed=True, flavor=flavor.to_dict(computed=False))