from __future__ import absolute_import, division, print_function
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.cisco.iosxr.plugins.module_utils.network.iosxr.facts.facts import Facts
from ansible_collections.cisco.iosxr.plugins.module_utils.network.iosxr.utils.utils import (
def _set_commands(self, want_ace, have_ace):
    """A helped method that checks if there is
           a delta between the `have_ace` and `want_ace`.
           If there is a delta then it calls `_compute_commands`
           to create the ACE line.

        :rtype: A string
        :returns: An ACE generated from a structured ACE dictionary
                  via a call to `_compute_commands`
        """
    if 'line' in want_ace:
        if want_ace['line'] != have_ace.get('line'):
            return self._compute_commands(want_ace)
    else:
        if 'prefix' in want_ace.get('source', {}) or 'prefix' in want_ace.get('destination', {}):
            self._prepare_for_diff(want_ace)
        protocol_opt_delta = {}
        delta = dict_diff(have_ace, want_ace)
        if want_ace.get('protocol_options', {}):
            protocol_opt_delta = set(flatten_dict(have_ace.get('protocol_options', {}))) ^ set(flatten_dict(want_ace.get('protocol_options', {})))
        if delta or protocol_opt_delta:
            if self.state not in ['replaced']:
                want_ace = self._dict_merge(have_ace, want_ace)
            return self._compute_commands(want_ace)