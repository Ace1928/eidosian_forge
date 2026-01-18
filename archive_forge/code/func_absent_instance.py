from __future__ import absolute_import, division, print_function
import base64
from ansible.module_utils._text import to_bytes, to_text
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (AnsibleCloudStack, cs_argument_spec,
def absent_instance(self):
    instance = self.get_instance()
    if instance:
        if instance['state'].lower() not in ['expunging', 'destroying', 'destroyed']:
            self.result['changed'] = True
            if not self.module.check_mode:
                res = self.query_api('destroyVirtualMachine', id=instance['id'])
                poll_async = self.module.params.get('poll_async')
                if poll_async:
                    instance = self.poll_job(res, 'virtualmachine')
    return instance