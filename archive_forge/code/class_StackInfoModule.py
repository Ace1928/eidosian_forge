from ansible_collections.openstack.cloud.plugins.module_utils.openstack import OpenStackModule
class StackInfoModule(OpenStackModule):
    argument_spec = dict(name=dict(), owner=dict(aliases=['owner_id']), project=dict(aliases=['project_id']), status=dict())
    module_kwargs = dict(supports_check_mode=True)

    def run(self):
        kwargs = {}
        owner_name_or_id = self.params['owner']
        if owner_name_or_id:
            owner = self.conn.orchestration.find_stack(owner_name_or_id)
            if owner:
                kwargs['owner_id'] = owner['id']
            else:
                self.exit_json(changed=False, stacks=[])
        project_name_or_id = self.params['project']
        if project_name_or_id:
            project = self.conn.identity.find_project(project_name_or_id)
            if project:
                kwargs['project_id'] = project['id']
            else:
                self.exit_json(changed=False, stacks=[])
        for k in ['name', 'status']:
            if self.params[k] is not None:
                kwargs[k] = self.params[k]
        stacks = [stack.to_dict(computed=False) for stack in self.conn.orchestration.stacks(**kwargs)]
        self.exit_json(changed=False, stacks=stacks)