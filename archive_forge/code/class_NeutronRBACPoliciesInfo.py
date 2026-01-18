from ansible_collections.openstack.cloud.plugins.module_utils.openstack import OpenStackModule
class NeutronRBACPoliciesInfo(OpenStackModule):
    argument_spec = dict(action=dict(choices=['access_as_external', 'access_as_shared']), object_id=dict(), object_type=dict(choices=['security_group', 'qos_policy', 'network']), policy_id=dict(), project=dict(aliases=['project_id']), target_project_id=dict())
    module_kwargs = dict(mutually_exclusive=[('object_id', 'object_type')], supports_check_mode=True)

    def run(self):
        project_name_or_id = self.params['project']
        project = None
        if project_name_or_id is not None:
            project = self.conn.identity.find_project(project_name_or_id)
            if not project:
                self.exit_json(changed=False, rbac_policies=[], policies=[])
        policy_id = self.params['policy_id']
        if policy_id:
            policy = self.conn.network.find_rbac_policy(policy_id)
            policies = [policy] if policy else []
        else:
            kwargs = dict(((k, self.params[k]) for k in ['action', 'object_type'] if self.params[k] is not None))
            if project:
                kwargs['project_id'] = project.id
            policies = list(self.conn.network.rbac_policies(**kwargs))
        for k in ['object_id', 'target_project_id']:
            if self.params[k] is not None:
                policies = [p for p in policies if p[k] == self.params[k]]
        if project:
            policies = [p for p in policies if p['location']['project']['id'] == project.id]
        policies = [p.to_dict(computed=False) for p in policies]
        self.exit_json(changed=False, rbac_policies=policies, policies=policies)