import time
from oslo_utils import timeutils
from oslo_utils import uuidutils
from tempest.lib import exceptions
from novaclient.tests.functional import base
def _test_cmd_with_not_existing_instance(self, cmd, args):
    try:
        self.nova('%s %s' % (cmd, args))
    except exceptions.CommandFailed as e:
        self.assertIn('ERROR (NotFound):', str(e))
    else:
        self.fail('%s is not failed on non existing instance.' % cmd)