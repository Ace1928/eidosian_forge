import os
import fixtures
from tempest.lib.common.utils import data_utils
from openstackclient.tests.functional import base
def _create_dummy_idp(self, add_clean_up=True):
    identity_provider = data_utils.rand_name('IdentityProvider')
    description = data_utils.rand_name('description')
    raw_output = self.openstack('identity provider create  %(name)s --description %(description)s --enable ' % {'name': identity_provider, 'description': description})
    if add_clean_up:
        self.addCleanup(self.openstack, 'identity provider delete %s' % identity_provider)
    items = self.parse_show(raw_output)
    self.assert_show_fields(items, self.IDENTITY_PROVIDER_FIELDS)
    return identity_provider