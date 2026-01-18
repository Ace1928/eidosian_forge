from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.cisco.iosxr.plugins.module_utils.network.iosxr.facts.facts import Facts
from ansible_collections.cisco.iosxr.plugins.module_utils.network.iosxr.iosxr import get_os_version
from ansible_collections.cisco.iosxr.plugins.module_utils.network.iosxr.utils.utils import (
def _render_interface_del_commands(self, want, have):
    """The command generator for delete commands
            w.r.t member interfaces
        :rtype: A list
        :returns: the commands necessary to update member
                  interfaces
        """
    commands = []
    if not want:
        want = {}
    have_members = have.get('members')
    if have_members:
        have_members = param_list_to_dict(deepcopy(have_members), unique_key='member')
        want_members = param_list_to_dict(deepcopy(want).get('members', []), unique_key='member')
        for key in have_members:
            if key not in want_members:
                member_cmd = ['no bundle id']
                pad_commands(member_cmd, key)
                commands.extend(member_cmd)
    return commands