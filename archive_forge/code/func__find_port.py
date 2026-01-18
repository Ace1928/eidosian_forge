from ansible_collections.openstack.cloud.plugins.module_utils.openstack import (
def _find_port(self):
    id = self.params['id']
    if id:
        return self.conn.baremetal.get_port(id)
    address = self.params['address']
    if address:
        ports = list(self.conn.baremetal.ports(address=address, details=True))
        if len(ports) == 1:
            return ports[0]
        elif len(ports) > 1:
            raise ValueError('Multiple ports with address {address} found. A ID must be defined in order to identify a unique port.'.format(address=address))
        else:
            return None
    raise AssertionError('id or address must be specified')