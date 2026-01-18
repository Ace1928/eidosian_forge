from ansible_collections.openstack.cloud.plugins.module_utils.openstack import OpenStackModule
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