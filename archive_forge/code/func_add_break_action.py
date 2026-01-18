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
def add_break_action(self, actions, current):
    if current and self.parameters['relationship_state'] == 'broken':
        if current['mirror_state'] == 'uninitialized':
            self.module.fail_json(msg='SnapMirror relationship cannot be broken if mirror state is uninitialized')
        elif current['relationship_type'] in ['load_sharing', 'vault']:
            self.module.fail_json(msg='SnapMirror break is not allowed in a load_sharing or vault relationship')
        elif current['mirror_state'] not in ['broken-off', 'broken_off']:
            actions.append('break')
            self.na_helper.changed = True