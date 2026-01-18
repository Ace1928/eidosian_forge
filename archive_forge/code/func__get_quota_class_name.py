from tempest.lib import exceptions
from novaclient.tests.functional import base
def _get_quota_class_name(self):
    """Returns a fake quota class name specific to this test class."""
    return 'fake-class-%s' % self.COMPUTE_API_VERSION.replace('.', '-')