from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flashblade.plugins.module_utils.purefb import (
from datetime import datetime
def _millisecs_to_time(millisecs):
    if millisecs:
        return (str(int(millisecs / 3600000 % 24)).zfill(2) + ':00',)
    return None