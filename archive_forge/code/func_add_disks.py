from __future__ import absolute_import, division, print_function
import re
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
import ansible_collections.netapp.ontap.plugins.module_utils.rest_response_helpers as rrh
def add_disks(self, count=0, disks=None, mirror_disks=None, disk_size=0, disk_size_with_unit=None):
    """
        Add additional disks to aggregate.
        :return: None
        """
    if self.use_rest:
        return self.add_disks_rest(count, disks, mirror_disks, disk_size, disk_size_with_unit)
    options = {'aggregate': self.parameters['name']}
    if count:
        options['disk-count'] = str(count)
    if disks and self.parameters.get('ignore_pool_checks'):
        options['ignore-pool-checks'] = str(self.parameters['ignore_pool_checks'])
    if disk_size:
        options['disk-size'] = str(disk_size)
    if disk_size_with_unit:
        options['disk-size-with-unit'] = disk_size_with_unit
    if self.parameters.get('disk_class'):
        options['disk-class'] = self.parameters['disk_class']
    if self.parameters.get('disk_type'):
        options['disk-type'] = self.parameters['disk_type']
    aggr_add = netapp_utils.zapi.NaElement.create_node_with_children('aggr-add', **options)
    if disks:
        aggr_add.add_child_elem(self.get_disks_or_mirror_disks_object('disks', disks))
    if mirror_disks:
        aggr_add.add_child_elem(self.get_disks_or_mirror_disks_object('mirror-disks', mirror_disks))
    try:
        self.server.invoke_successfully(aggr_add, enable_tunneling=True)
    except netapp_utils.zapi.NaApiError as error:
        self.module.fail_json(msg='Error adding additional disks to aggregate %s: %s' % (self.parameters['name'], to_native(error)), exception=traceback.format_exc())