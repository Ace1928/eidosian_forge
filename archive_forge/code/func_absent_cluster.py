from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (
def absent_cluster(self):
    cluster = self.get_cluster()
    if cluster:
        self.result['changed'] = True
        args = {'id': cluster['id']}
        if not self.module.check_mode:
            self.query_api('deleteCluster', **args)
    return cluster