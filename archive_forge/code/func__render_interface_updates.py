from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.cisco.iosxr.plugins.module_utils.network.iosxr.facts.facts import Facts
from ansible_collections.cisco.iosxr.plugins.module_utils.network.iosxr.iosxr import get_os_version
from ansible_collections.cisco.iosxr.plugins.module_utils.network.iosxr.utils.utils import (
def _render_interface_updates(self, want, have):
    """The command generator for updates to member
            interfaces
        :rtype: A list
        :returns: the commands necessary to update member
                  interfaces
        """
    commands = []
    if not have:
        have = {'name': want['name']}
    member_diff = diff_list_of_dicts(want['members'], have.get('members', []))
    for diff in member_diff:
        diff_cmd = []
        bundle_cmd = 'bundle id {0}'.format(want['name'].split('Bundle-Ether')[1])
        if diff.get('mode'):
            bundle_cmd += ' mode {0}'.format(diff.get('mode'))
        diff_cmd.append(bundle_cmd)
        pad_commands(diff_cmd, diff['member'])
        commands.extend(diff_cmd)
    return commands