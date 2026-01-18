from __future__ import absolute_import, division, print_function
import re
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def _tmplt_area_authentication(area):
    command = 'area {area_id} authentication'.format(**area)
    if area.get('authentication', {}).get('message_digest'):
        command += ' message-digest'
    return command