from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic, rest_vserver
def _fill_in_access_policy(self, access_policy_obj):
    if self.parameters.get('is_scan_mandatory') is not None:
        access_policy_obj.add_new_child('is-scan-mandatory', str(self.parameters['is_scan_mandatory']).lower())
    if self.parameters.get('max_file_size'):
        access_policy_obj.add_new_child('max-file-size', str(self.parameters['max_file_size']))
    if self.parameters.get('scan_files_with_no_ext') is not None:
        access_policy_obj.add_new_child('scan-files-with-no-ext', str(self.parameters['scan_files_with_no_ext']))
    if 'file_ext_to_exclude' in self.parameters:
        ext_obj = netapp_utils.zapi.NaElement('file-ext-to-exclude')
        access_policy_obj.add_child_elem(ext_obj)
        if len(self.parameters['file_ext_to_exclude']) < 1:
            ext_obj.add_new_child('file-extension', '')
        else:
            for extension in self.parameters['file_ext_to_exclude']:
                ext_obj.add_new_child('file-extension', extension)
    if 'file_ext_to_include' in self.parameters:
        ext_obj = netapp_utils.zapi.NaElement('file-ext-to-include')
        access_policy_obj.add_child_elem(ext_obj)
        for extension in self.parameters['file_ext_to_include']:
            ext_obj.add_new_child('file-extension', extension)
    if 'filters' in self.parameters:
        ui_filter_obj = netapp_utils.zapi.NaElement('filters')
        access_policy_obj.add_child_elem(ui_filter_obj)
        if len(self.parameters['filters']) < 1:
            ui_filter_obj.add_new_child('vscan-on-access-policy-ui-filter', '')
        else:
            for filter in self.parameters['filters']:
                ui_filter_obj.add_new_child('vscan-on-access-policy-ui-filter', filter)
    if 'paths_to_exclude' in self.parameters:
        path_obj = netapp_utils.zapi.NaElement('paths-to-exclude')
        access_policy_obj.add_child_elem(path_obj)
        if len(self.parameters['paths_to_exclude']) < 1:
            path_obj.add_new_child('file-path', '')
        else:
            for path in self.parameters['paths_to_exclude']:
                path_obj.add_new_child('file-path', path)
    return access_policy_obj