from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import raise_from
from ..module_utils.cloudstack import (AnsibleCloudStack, cs_argument_spec,
def _cloudstack_ver(self):
    capabilities = self.get_capabilities()
    return LooseVersion(capabilities['cloudstackversion'])