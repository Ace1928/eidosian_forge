from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def _tmplt_ospf_area_nssa_translate(config_data):
    if 'nssa' in config_data and 'translate' in config_data['nssa']:
        command = 'area {area_id} nssa'.format(**config_data)
        if 'translate' in config_data['nssa']:
            command += ' translate type7 {translate}'.format(**config_data['nssa'])
        return command