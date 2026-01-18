from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
def _char_data(self, data):
    """ Dump XML elemet data """
    self.result_dict['xml_dict'][str(self.result_dict['xml_dict']['active_element'])]['data'] = repr(data)