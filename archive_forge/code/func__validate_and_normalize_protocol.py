from __future__ import absolute_import, division, print_function
from ast import literal_eval
from ansible.module_utils._text import to_text
from ansible.module_utils.common.validation import check_required_arguments
from ansible.module_utils.connection import ConnectionError
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import Facts
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.formatted_diff_utils import (
def _validate_and_normalize_protocol(self, acl_type, acl_name, rule):
    protocol = rule.get('protocol')
    if protocol:
        if protocol.get('number') is not None:
            if protocol['number'] in protocol_number_to_name_map:
                protocol['name'] = protocol_number_to_name_map[protocol.pop('number')]
        protocol_name = protocol.get('name')
        if acl_type == 'ipv4' and protocol_name in ('ipv6', 'icmpv6') or (acl_type == 'ipv6' and protocol_name in ('ip', 'icmp')):
            self._invalid_rule('invalid protocol {0} for {1} ACL'.format(protocol_name, acl_type), acl_type, acl_name, rule['sequence_num'])