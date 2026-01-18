from __future__ import absolute_import, division, print_function
import re
from copy import deepcopy
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.facts.facts import Facts
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.utils.utils import (
def _add_icmp(self, attr, w, h, cmd, opr):
    """
        This function forms the commands for 'icmp' attributes based on the 'opr'.
        :param attr: attribute name.
        :param w: base config.
        :param h: target config.
        :param cmd: commands to be prepend.
        :return: generated list of commands.
        """
    commands = []
    h_icmp = {}
    l_set = ('code', 'type', 'type_name')
    if w[attr]:
        if h and attr in h.keys():
            h_icmp = h.get(attr) or {}
        for item, val in iteritems(w[attr]):
            if opr and item in l_set and (not (h_icmp and self._is_w_same(w[attr], h_icmp, item))):
                if item == 'type_name':
                    os_version = self._get_os_version()
                    ver = re.search('vyos ([\\d\\.]+)-?.*', os_version, re.IGNORECASE)
                    if ver.group(1) >= '1.4':
                        param_name = 'type-name'
                    else:
                        param_name = 'type'
                    if 'ipv6-name' in cmd:
                        commands.append(cmd + (' ' + 'icmpv6' + ' ' + param_name + ' ' + val))
                    else:
                        commands.append(cmd + (' ' + attr + ' ' + item.replace('_', '-') + ' ' + val))
                else:
                    commands.append(cmd + (' ' + attr + ' ' + item + ' ' + str(val)))
            elif not opr and item in l_set and (not (h_icmp and self._in_target(h_icmp, item))):
                commands.append(cmd + (' ' + attr + ' ' + item))
    return commands