from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.facts.facts import Facts
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.utils.utils import (
def _add_blackhole(self, key, want, have):
    """
        This function gets the diff for blackhole config specific attributes
        and form the commands for attributes which are present in want but not in have.
        :param key:
        :param want:
        :param have:
        :return: list of commands
        """
    commands = []
    want_copy = deepcopy(remove_empties(want))
    have_copy = deepcopy(remove_empties(have))
    want_blackhole = want_copy.get(key) or {}
    have_blackhole = have_copy.get(key) or {}
    updates = dict_delete(want_blackhole, have_blackhole)
    if updates:
        for attrib, value in iteritems(updates):
            if value:
                if attrib == 'distance':
                    commands.append(self._compute_command(dest=want['dest'], key='blackhole', attrib=attrib, remove=False, value=str(value)))
                elif attrib == 'type':
                    commands.append(self._compute_command(dest=want['dest'], key='blackhole'))
    return commands