from ansible_collections.openstack.cloud.plugins.module_utils.openstack import OpenStackModule
class COEClusterModule(OpenStackModule):
    argument_spec = dict(cluster_template_id=dict(), discovery_url=dict(), flavor_id=dict(), is_floating_ip_enabled=dict(type='bool', aliases=['floating_ip_enabled']), keypair=dict(no_log=False), labels=dict(type='raw'), master_count=dict(type='int'), master_flavor_id=dict(), name=dict(required=True), node_count=dict(type='int'), state=dict(default='present', choices=['absent', 'present']))
    module_kwargs = dict(required_if=[('state', 'present', ('cluster_template_id',))], supports_check_mode=True)

    def run(self):
        state = self.params['state']
        cluster = self._find()
        if self.ansible.check_mode:
            self.exit_json(changed=self._will_change(state, cluster))
        if state == 'present' and (not cluster):
            cluster = self._create()
            self.exit_json(changed=True, cluster=cluster.to_dict(computed=False))
        elif state == 'present' and cluster:
            update = self._build_update(cluster)
            if update:
                cluster = self._update(cluster, update)
            self.exit_json(changed=bool(update), cluster=cluster.to_dict(computed=False))
        elif state == 'absent' and cluster:
            self._delete(cluster)
            self.exit_json(changed=True)
        elif state == 'absent' and (not cluster):
            self.exit_json(changed=False)

    def _build_update(self, cluster):
        update = {}
        non_updateable_keys = [k for k in ['cluster_template_id', 'discovery_url', 'flavor_id', 'is_floating_ip_enabled', 'keypair', 'master_count', 'master_flavor_id', 'name', 'node_count'] if self.params[k] is not None and self.params[k] != cluster[k]]
        labels = self.params['labels']
        if labels is not None:
            if isinstance(labels, str):
                labels = dict([tuple(kv.split(':')) for kv in labels.split(',')])
            if labels != cluster['labels']:
                non_updateable_keys.append('labels')
        if non_updateable_keys:
            self.fail_json(msg='Cannot update parameters {0}'.format(non_updateable_keys))
        attributes = dict(((k, self.params[k]) for k in [] if self.params[k] is not None and self.params[k] != cluster[k]))
        if attributes:
            update['attributes'] = attributes
        return update

    def _create(self):
        kwargs = dict(((k, self.params[k]) for k in ['cluster_template_id', 'discovery_url', 'flavor_id', 'is_floating_ip_enabled', 'keypair', 'master_count', 'master_flavor_id', 'name', 'node_count'] if self.params[k] is not None))
        labels = self.params['labels']
        if labels is not None:
            if isinstance(labels, str):
                labels = dict([tuple(kv.split(':')) for kv in labels.split(',')])
            kwargs['labels'] = labels
        kwargs['create_timeout'] = self.params['timeout']
        cluster = self.conn.container_infrastructure_management.create_cluster(**kwargs)
        if not self.params['wait']:
            return cluster
        if self.params['wait']:
            cluster = self.sdk.resource.wait_for_status(self.conn.container_infrastructure_management, cluster, status='active', failures=['error'], wait=self.params['timeout'])
        return cluster

    def _delete(self, cluster):
        self.conn.container_infrastructure_management.delete_cluster(cluster['id'])
        if self.params['wait']:
            self.sdk.resource.wait_for_delete(self.conn.container_infrastructure_management, cluster, interval=None, wait=self.params['timeout'])

    def _find(self):
        name = self.params['name']
        filters = {}
        cluster_template_id = self.params['cluster_template_id']
        if cluster_template_id is not None:
            filters['cluster_template_id'] = cluster_template_id
        return self.conn.get_coe_cluster(name_or_id=name, filters=filters)

    def _update(self, cluster, update):
        attributes = update.get('attributes')
        if attributes:
            pass
        return cluster

    def _will_change(self, state, cluster):
        if state == 'present' and (not cluster):
            return True
        elif state == 'present' and cluster:
            return bool(self._build_update(cluster))
        elif state == 'absent' and cluster:
            return True
        else:
            return False