from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (AnsibleCloudStack, cs_argument_spec,
def _get_common_configuration_args(self):
    args = {'name': self.module.params.get('name'), 'accountid': self.get_account(key='id'), 'storageid': self.get_storage(key='id'), 'zoneid': self.get_zone(key='id'), 'clusterid': self.get_cluster(key='id')}
    return args