from ansible_collections.openstack.cloud.plugins.module_utils.openstack import OpenStackModule
def _filter_ips(self, server):

    def _flatten(lists):
        return [item for sublist in lists for item in sublist]
    if server['addresses'] is None:
        server = self.conn.compute.get_server(server)
    if not server['addresses']:
        return []
    return [address['addr'] for address in _flatten(server['addresses'].values()) if address['OS-EXT-IPS:type'] == 'floating']