from ansible_collections.openstack.cloud.plugins.module_utils.openstack import OpenStackModule
from ansible_collections.openstack.cloud.plugins.module_utils.resource import StateMachine
class IdentityUserModule(OpenStackModule):
    argument_spec = dict(default_project=dict(), description=dict(), domain=dict(), email=dict(), is_enabled=dict(default=True, type='bool', aliases=['enabled']), name=dict(required=True), password=dict(no_log=True), state=dict(default='present', choices=['absent', 'present']), update_password=dict(default='on_create', choices=['always', 'on_create']))
    module_kwargs = dict()

    class _StateMachine(StateMachine):

        def _build_update(self, resource, attributes, updateable_attributes, non_updateable_attributes, update_password='on_create', **kwargs):
            if update_password == 'always' and 'password' not in attributes:
                self.ansible.fail_json(msg="update_password is 'always' but password is missing")
            elif update_password == 'on_create' and 'password' in attributes:
                attributes.pop('password')
            return super()._build_update(resource, attributes, updateable_attributes, non_updateable_attributes, **kwargs)

        def _find(self, attributes, **kwargs):
            query_args = dict(((k, attributes[k]) for k in ['domain_id'] if k in attributes and attributes[k] is not None))
            return self.find_function(attributes['name'], **query_args)

    def run(self):
        sm = self._StateMachine(connection=self.conn, service_name='identity', type_name='user', sdk=self.sdk, ansible=self.ansible)
        kwargs = dict(((k, self.params[k]) for k in ['state', 'timeout', 'update_password'] if self.params[k] is not None))
        kwargs['attributes'] = dict(((k, self.params[k]) for k in ['description', 'email', 'is_enabled', 'name', 'password'] if self.params[k] is not None))
        domain_name_or_id = self.params['domain']
        if domain_name_or_id is not None:
            domain = self.conn.identity.find_domain(domain_name_or_id, ignore_missing=False)
            kwargs['attributes']['domain_id'] = domain.id
        default_project_name_or_id = self.params['default_project']
        if default_project_name_or_id is not None:
            query_args = dict(((k, kwargs['attributes'][k]) for k in ['domain_id'] if k in kwargs['attributes'] and kwargs['attributes'][k] is not None))
            project = self.conn.identity.find_project(default_project_name_or_id, ignore_missing=False, **query_args)
            kwargs['attributes']['default_project_id'] = project.id
        user, is_changed = sm(check_mode=self.ansible.check_mode, updateable_attributes=None, non_updateable_attributes=['domain_id'], wait=False, **kwargs)
        if user is None:
            self.exit_json(changed=is_changed)
        else:
            self.exit_json(changed=is_changed, user=user.to_dict(computed=False))