from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.cisco.iosxr.plugins.module_utils.network.iosxr.facts.facts import Facts
from ansible_collections.cisco.iosxr.plugins.module_utils.network.iosxr.iosxr import get_os_version
from ansible_collections.cisco.iosxr.plugins.module_utils.network.iosxr.utils.utils import (
def _render_bundle_updates(self, want, have):
    """The command generator for updates to bundles
         :rtype: A list
        :returns: the commands necessary to update bundles
        """
    commands = []
    if not have:
        have = {'name': want['name']}
    want_copy = deepcopy(want)
    have_copy = deepcopy(have)
    want_copy.pop('members', [])
    have_copy.pop('members', [])
    bundle_updates = dict_diff(have_copy, want_copy)
    if bundle_updates:
        for key, value in iteritems(flatten_dict(remove_empties(bundle_updates))):
            commands.append(self._compute_commands(key=key, value=value))
    return commands