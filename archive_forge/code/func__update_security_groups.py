from ansible_collections.openstack.cloud.plugins.module_utils.openstack import OpenStackModule
import copy
def _update_security_groups(self, server, update):
    add_security_groups = update.get('add_security_groups')
    if add_security_groups:
        for sg in add_security_groups:
            self.conn.compute.add_security_group_to_server(server, sg)
    remove_security_groups = update.get('remove_security_groups')
    if remove_security_groups:
        for sg in remove_security_groups:
            self.conn.compute.remove_security_group_from_server(server, sg)
    return server