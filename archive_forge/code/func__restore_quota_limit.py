import uuid
from tempest.lib.common.utils import data_utils
from tempest.lib import exceptions
from openstackclient.tests.functional import base
def _restore_quota_limit(self, resource, limit, project):
    self.openstack('quota set --%s %s %s' % (resource, limit, project))