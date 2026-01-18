from __future__ import absolute_import, division, print_function
import re
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_elementsw_module import NaElementSWModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def add_active_actions(self, actions, current):
    if current and self.parameters['relationship_state'] == 'active':
        if self.parameters['initialize'] and current['mirror_state'] == 'uninitialized' and (current['current_transfer_type'] != 'initialize'):
            actions.append('initialize')
            self.na_helper.changed = True
        if current['status'] == 'quiesced' or current['mirror_state'] == 'paused':
            actions.append('resume')
            self.na_helper.changed = True
        if current['mirror_state'] in ['broken-off', 'broken_off']:
            actions.append('resync')
            self.na_helper.changed = True
        elif self.parameters['update']:
            actions.append('check_for_update')