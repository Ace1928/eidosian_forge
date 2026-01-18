from ansible_collections.openstack.cloud.plugins.module_utils.openstack import OpenStackModule
import copy
def _build_update_security_groups(self, server):
    update = {}
    required_security_groups = dict(((sg['id'], sg) for sg in [self.conn.network.find_security_group(security_group_name_or_id, ignore_missing=False) for security_group_name_or_id in self.params['security_groups']]))
    server = self.conn.compute.fetch_server_security_groups(server)
    assigned_security_groups = dict(((sg['id'], self.conn.network.get_security_group(sg['id'])) for sg in server.security_groups))
    add_security_groups = [sg for sg_id, sg in required_security_groups.items() if sg_id not in assigned_security_groups]
    if add_security_groups:
        update['add_security_groups'] = add_security_groups
    remove_security_groups = [sg for sg_id, sg in assigned_security_groups.items() if sg_id not in required_security_groups]
    if remove_security_groups:
        update['remove_security_groups'] = remove_security_groups
    return update