from ansible_collections.openstack.cloud.plugins.module_utils.openstack import OpenStackModule
class IdentityGroupAssignment(OpenStackModule):
    argument_spec = dict(group=dict(required=True), state=dict(default='present', choices=['absent', 'present']), user=dict(required=True))
    module_kwargs = dict(supports_check_mode=True)

    def run(self):
        user_name_or_id = self.params['user']
        user = self.conn.identity.find_user(user_name_or_id, ignore_missing=False)
        group_name_or_id = self.params['group']
        group = self.conn.identity.find_group(group_name_or_id, ignore_missing=False)
        is_user_in_group = self.conn.identity.check_user_in_group(user, group)
        state = self.params['state']
        if self.ansible.check_mode:
            self.exit_json(changed=state == 'present' and (not is_user_in_group) or (state == 'absent' and is_user_in_group))
        if state == 'present' and (not is_user_in_group):
            self.conn.identity.add_user_to_group(user, group)
            self.exit_json(changed=True)
        elif state == 'absent' and is_user_in_group:
            self.conn.identity.remove_user_from_group(user, group)
            self.exit_json(changed=True)
        else:
            self.exit_json(changed=False)