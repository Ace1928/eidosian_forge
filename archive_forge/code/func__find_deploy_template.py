from ansible_collections.openstack.cloud.plugins.module_utils.openstack import (
def _find_deploy_template(self):
    id_or_name = self.params['id'] if self.params['id'] else self.params['name']
    try:
        return self.conn.baremetal.get_deploy_template(id_or_name)
    except self.sdk.exceptions.ResourceNotFound:
        return None