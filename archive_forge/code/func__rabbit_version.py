from __future__ import absolute_import, division, print_function
import json
import re
from ansible_collections.community.rabbitmq.plugins.module_utils.version import LooseVersion as Version
from ansible.module_utils.basic import AnsibleModule
def _rabbit_version(self):
    status = self._exec(['status'], True, False, False)
    version_match = re.search('{rabbit,".*","(?P<version>.*)"}', status)
    if version_match:
        return Version(version_match.group('version'))
    version_match = re.search('RabbitMQ version: (?P<version>.*)', status)
    if version_match:
        return Version(version_match.group('version'))
    return None