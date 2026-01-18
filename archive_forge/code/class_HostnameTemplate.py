from __future__ import absolute_import, division, print_function
import re
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
class HostnameTemplate(NetworkTemplate):

    def __init__(self, lines=None, module=None):
        prefix = {'set': 'set', 'remove': 'delete'}
        super(HostnameTemplate, self).__init__(lines=lines, tmplt=self, prefix=prefix, module=module)
    PARSERS = [{'name': 'hostname', 'getval': re.compile('\n                ^set\\ssystem\\shost-name\n                \\s+(?P<name>\\S+)\n                $', re.VERBOSE), 'setval': 'system host-name {{ hostname }}', 'result': {'hostname': '{{ name }}'}}]