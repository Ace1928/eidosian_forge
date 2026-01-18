import os
import fixtures
from tempest.lib.common.utils import data_utils
from openstackclient.tests.functional import base
def _create_dummy_region(self, parent_region=None, add_clean_up=True):
    region_id = data_utils.rand_name('TestRegion')
    description = data_utils.rand_name('description')
    parent_region_arg = ''
    if parent_region is not None:
        parent_region_arg = '--parent-region %s' % parent_region
    raw_output = self.openstack('region create %(parent_region_arg)s --description %(description)s %(id)s' % {'parent_region_arg': parent_region_arg, 'description': description, 'id': region_id})
    if add_clean_up:
        self.addCleanup(self.openstack, 'region delete %s' % region_id)
    items = self.parse_show(raw_output)
    self.assert_show_fields(items, self.REGION_FIELDS)
    return region_id