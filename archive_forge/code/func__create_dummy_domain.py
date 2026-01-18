import os
import fixtures
from tempest.lib.common.utils import data_utils
from openstackclient.tests.functional import base
def _create_dummy_domain(self, add_clean_up=True):
    domain_name = data_utils.rand_name('TestDomain')
    domain_description = data_utils.rand_name('description')
    self.openstack('domain create --description %(description)s --enable %(name)s' % {'description': domain_description, 'name': domain_name})
    if add_clean_up:
        self.addCleanup(self.openstack, 'domain delete %s' % domain_name)
        self.addCleanup(self.openstack, 'domain set --disable %s' % domain_name)
    return domain_name