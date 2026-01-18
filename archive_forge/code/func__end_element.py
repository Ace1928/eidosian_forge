from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
def _end_element(self, name):
    self.result_dict['xml_dict']['last_element'] = name
    self.result_dict['xml_dict']['active_element'] = ''