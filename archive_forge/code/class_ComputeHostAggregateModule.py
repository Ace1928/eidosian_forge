from ansible_collections.openstack.cloud.plugins.module_utils.openstack import OpenStackModule
class ComputeHostAggregateModule(OpenStackModule):
    argument_spec = dict(name=dict(required=True), metadata=dict(type='dict'), availability_zone=dict(), hosts=dict(type='list', elements='str'), purge_hosts=dict(default=True, type='bool'), state=dict(default='present', choices=['absent', 'present']))
    module_kwargs = dict(supports_check_mode=True)

    def _needs_update(self, aggregate):
        new_metadata = self.params['metadata'] or {}
        if self.params['availability_zone'] is not None:
            new_metadata['availability_zone'] = self.params['availability_zone']
        if self.params['hosts'] is not None:
            if self.params['purge_hosts']:
                if set(self.params['hosts']) != set(aggregate.hosts):
                    return True
            else:
                intersection = set(self.params['hosts']).intersection(set(aggregate.hosts))
                if set(self.params['hosts']) != intersection:
                    return True
        for param in ('availability_zone', 'metadata'):
            if self.params[param] is not None and self.params[param] != aggregate[param]:
                return True
        return False

    def _system_state_change(self, aggregate):
        state = self.params['state']
        if state == 'absent' and aggregate:
            return True
        if state == 'present':
            if aggregate is None:
                return True
            return self._needs_update(aggregate)
        return False

    def _update_hosts(self, aggregate, hosts, purge_hosts):
        if hosts is None:
            return
        hosts_to_add = set(hosts) - set(aggregate['hosts'] or [])
        for host in hosts_to_add:
            self.conn.compute.add_host_to_aggregate(aggregate.id, host)
        if not purge_hosts:
            return
        hosts_to_remove = set(aggregate['hosts'] or []) - set(hosts)
        for host in hosts_to_remove:
            self.conn.compute.remove_host_from_aggregate(aggregate.id, host)

    def run(self):
        name = self.params['name']
        metadata = self.params['metadata']
        availability_zone = self.params['availability_zone']
        hosts = self.params['hosts']
        purge_hosts = self.params['purge_hosts']
        state = self.params['state']
        if metadata is not None:
            metadata.pop('availability_zone', None)
        aggregate = self.conn.compute.find_aggregate(name)
        if self.ansible.check_mode:
            self.exit_json(changed=self._system_state_change(aggregate))
        changed = False
        if state == 'present':
            if aggregate is None:
                aggregate = self.conn.compute.create_aggregate(name=name, availability_zone=availability_zone)
                self._update_hosts(aggregate, hosts, False)
                if metadata:
                    self.conn.compute.set_aggregate_metadata(aggregate, metadata)
                changed = True
            elif self._needs_update(aggregate):
                if availability_zone is not None:
                    aggregate = self.conn.compute.update_aggregate(aggregate, name=name, availability_zone=availability_zone)
                if metadata is not None:
                    metas = metadata
                    for i in set(aggregate.metadata.keys() - set(metadata.keys())):
                        if i != 'availability_zone':
                            metas[i] = None
                    self.conn.compute.set_aggregate_metadata(aggregate, metas)
                self._update_hosts(aggregate, hosts, purge_hosts)
                changed = True
            aggregate = self.conn.compute.find_aggregate(name)
            if aggregate:
                aggregate = aggregate.to_dict(computed=False)
            self.exit_json(changed=changed, aggregate=aggregate)
        elif state == 'absent' and aggregate is not None:
            self._update_hosts(aggregate, [], True)
            self.conn.compute.delete_aggregate(aggregate.id)
            changed = True
        self.exit_json(changed=changed)