from __future__ import absolute_import, division, print_function
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.resource_module import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.facts.facts import Facts
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.rm_templates.ntp_global import (
def _serveroptions_list_to_dict(self, entry):
    serveroptions_dict = {}
    for Opk, Op in iteritems(entry):
        if Opk == 'options':
            for val in Op:
                dict = {}
                dict.update({'server': entry['server']})
                dict.update({Opk: val})
                serveroptions_dict.update({entry['server'] + '_' + val: dict})
    return serveroptions_dict