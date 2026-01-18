from __future__ import absolute_import, division, print_function
import copy
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def _create_get_volume_return(self, results, uuid=None):
    """
        Create a return value from volume-autosize-get info file
        :param results:
        :return:
        """
    return_value = {}
    if self.use_rest:
        return_value['uuid'] = uuid
        if 'mode' in results:
            return_value['mode'] = results['mode']
        if 'grow_threshold' in results:
            return_value['grow_threshold_percent'] = results['grow_threshold']
        if 'maximum' in results:
            return_value['maximum_size'] = results['maximum']
        if 'minimum' in results:
            return_value['minimum_size'] = results['minimum']
        if 'shrink_threshold' in results:
            return_value['shrink_threshold_percent'] = results['shrink_threshold']
    else:
        if results.get_child_by_name('mode'):
            return_value['mode'] = results.get_child_content('mode')
        if results.get_child_by_name('grow-threshold-percent'):
            return_value['grow_threshold_percent'] = int(results.get_child_content('grow-threshold-percent'))
        if results.get_child_by_name('increment-size'):
            return_value['increment_size'] = results.get_child_content('increment-size')
        if results.get_child_by_name('maximum-size'):
            return_value['maximum_size'] = results.get_child_content('maximum-size')
        if results.get_child_by_name('minimum-size'):
            return_value['minimum_size'] = results.get_child_content('minimum-size')
        if results.get_child_by_name('shrink-threshold-percent'):
            return_value['shrink_threshold_percent'] = int(results.get_child_content('shrink-threshold-percent'))
    if not return_value:
        return_value = None
    return return_value