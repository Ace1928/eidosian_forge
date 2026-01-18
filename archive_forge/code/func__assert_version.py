from __future__ import absolute_import, division, print_function
import json
import re
from ansible_collections.community.rabbitmq.plugins.module_utils.version import LooseVersion as Version
from ansible.module_utils.basic import AnsibleModule
def _assert_version(self):
    if self._version and self._version < Version('3.8.10'):
        self._module.fail_json(changed=False, msg='User limits are only available for RabbitMQ >= 3.8.10. Detected version: %s' % self._version)