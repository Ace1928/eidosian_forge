import random
import string
from tempest.lib import decorators
from novaclient.tests.functional import base
from novaclient.tests.functional.v2.legacy import test_servers
from novaclient.v2 import shell
def _setup_extra_specs(self, flavor_id):
    extra_spec_key = 'dummykey'
    self.nova('flavor-key', params='%(flavor)s set %(key)s=dummyval' % {'flavor': flavor_id, 'key': extra_spec_key})
    unset_params = '%(flavor)s unset %(key)s' % {'flavor': flavor_id, 'key': extra_spec_key}
    self.addCleanup(self.nova, 'flavor-key', params=unset_params)