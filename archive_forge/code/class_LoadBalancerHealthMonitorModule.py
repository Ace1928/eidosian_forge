from ansible_collections.openstack.cloud.plugins.module_utils.openstack import OpenStackModule
class LoadBalancerHealthMonitorModule(OpenStackModule):
    argument_spec = dict(delay=dict(type='int'), expected_codes=dict(), health_monitor_timeout=dict(type='int', aliases=['resp_timeout']), http_method=dict(), is_admin_state_up=dict(type='bool', aliases=['admin_state_up']), max_retries=dict(type='int'), max_retries_down=dict(type='int'), name=dict(required=True), pool=dict(), state=dict(default='present', choices=['absent', 'present']), type=dict(default='HTTP'), url_path=dict())
    module_kwargs = dict(required_if=[('state', 'present', ('delay', 'health_monitor_timeout', 'max_retries', 'pool'))], supports_check_mode=True)

    def run(self):
        state = self.params['state']
        health_monitor = self._find()
        if self.ansible.check_mode:
            self.exit_json(changed=self._will_change(state, health_monitor))
        if state == 'present' and (not health_monitor):
            health_monitor = self._create()
            self.exit_json(changed=True, health_monitor=health_monitor.to_dict(computed=False))
        elif state == 'present' and health_monitor:
            update = self._build_update(health_monitor)
            if update:
                health_monitor = self._update(health_monitor, update)
            self.exit_json(changed=bool(update), health_monitor=health_monitor.to_dict(computed=False))
        elif state == 'absent' and health_monitor:
            self._delete(health_monitor)
            self.exit_json(changed=True)
        elif state == 'absent' and (not health_monitor):
            self.exit_json(changed=False)

    def _build_update(self, health_monitor):
        update = {}
        non_updateable_keys = [k for k in ['type'] if self.params[k] is not None and self.params[k] != health_monitor[k]]
        pool_name_or_id = self.params['pool']
        pool = self.conn.load_balancer.find_pool(name_or_id=pool_name_or_id, ignore_missing=False)
        if health_monitor['pools'] != [dict(id=pool.id)]:
            non_updateable_keys.append('pool')
        if non_updateable_keys:
            self.fail_json(msg='Cannot update parameters {0}'.format(non_updateable_keys))
        attributes = dict(((k, self.params[k]) for k in ['delay', 'expected_codes', 'http_method', 'is_admin_state_up', 'max_retries', 'max_retries_down', 'type', 'url_path'] if self.params[k] is not None and self.params[k] != health_monitor[k]))
        health_monitor_timeout = self.params['health_monitor_timeout']
        if health_monitor_timeout is not None and health_monitor_timeout != health_monitor['timeout']:
            attributes['timeout'] = health_monitor_timeout
        if attributes:
            update['attributes'] = attributes
        return update

    def _create(self):
        kwargs = dict(((k, self.params[k]) for k in ['delay', 'expected_codes', 'http_method', 'is_admin_state_up', 'max_retries', 'max_retries_down', 'name', 'type', 'url_path'] if self.params[k] is not None))
        health_monitor_timeout = self.params['health_monitor_timeout']
        if health_monitor_timeout is not None:
            kwargs['timeout'] = health_monitor_timeout
        pool_name_or_id = self.params['pool']
        pool = self.conn.load_balancer.find_pool(name_or_id=pool_name_or_id, ignore_missing=False)
        kwargs['pool_id'] = pool.id
        health_monitor = self.conn.load_balancer.create_health_monitor(**kwargs)
        if self.params['wait']:
            health_monitor = self.sdk.resource.wait_for_status(self.conn.load_balancer, health_monitor, status='active', failures=['error'], wait=self.params['timeout'], attribute='provisioning_status')
        return health_monitor

    def _delete(self, health_monitor):
        self.conn.load_balancer.delete_health_monitor(health_monitor.id)

    def _find(self):
        name = self.params['name']
        return self.conn.load_balancer.find_health_monitor(name_or_id=name)

    def _update(self, health_monitor, update):
        attributes = update.get('attributes')
        if attributes:
            health_monitor = self.conn.load_balancer.update_health_monitor(health_monitor.id, **attributes)
        if self.params['wait']:
            health_monitor = self.sdk.resource.wait_for_status(self.conn.load_balancer, health_monitor, status='active', failures=['error'], wait=self.params['timeout'], attribute='provisioning_status')
        return health_monitor

    def _will_change(self, state, health_monitor):
        if state == 'present' and (not health_monitor):
            return True
        elif state == 'present' and health_monitor:
            return bool(self._build_update(health_monitor))
        elif state == 'absent' and health_monitor:
            return True
        else:
            return False