from __future__ import absolute_import, division, print_function
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.cisco.iosxr.plugins.module_utils.network.iosxr.facts.facts import Facts
from ansible_collections.cisco.iosxr.plugins.module_utils.network.iosxr.utils.utils import (
def _prepare_for_diff(self, ace):
    """This method prepares the want ace dict
           for diff calculation against the have ace dict.

        :param ace: The want ace to prepare for diff calculation
        """
    for x in ['source', 'destination']:
        prefix = ace.get(x, {}).get('prefix')
        if prefix and is_ipv4_address(prefix):
            del ace[x]['prefix']
            ace[x]['address'], ace[x]['wildcard_bits'] = prefix_to_address_wildcard(prefix)